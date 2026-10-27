from flask import Flask, g, request, jsonify
from flask_cors import CORS
from datetime import date
import joblib
import math
import pandas as pd
import re
import shap
import threading
import unicodedata
from werkzeug.exceptions import RequestEntityTooLarge

import db
from bulk_upload import (
    BulkUploadError,
    MAX_BULK_UPLOAD_BYTES,
    parse_csv,
)
from student_registration import (
    MAX_REGISTRATION_BYTES,
    StudentRegistrationFileError,
    parse_registration_csv,
)
from auth import auth_blueprint, can_access_student, forbidden, require_auth

# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

# POST /auth/login, POST /auth/logout, GET /auth/me
app.register_blueprint(auth_blueprint)


@app.before_request
def limit_bulk_upload_request_size():

    if request.endpoint in {
        "validate_bulk_student_upload",
        "import_bulk_students",
        "validate_student_registration",
        "import_student_registration",
    }:
        request.max_content_length = max(
            MAX_BULK_UPLOAD_BYTES, MAX_REGISTRATION_BYTES
        ) + 128 * 1024


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):

    return jsonify({
        "status": "error",
        "message": "The upload exceeds the maximum request size."
    }), 413


# ============================================================
# LOAD MODEL AND DATASET
# ============================================================

model = joblib.load("final_student_performance_model.pkl")

df = pd.read_csv("data/Final_Marks_Data.csv")

features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

print("===== MODEL LOADED SUCCESSFULLY =====")


# ============================================================
# SHAP EXPLAINER
# ============================================================

# Float background (same values) so SHAP's masking buffer does not
# truncate decimal inputs to the dataset's integer dtype
masker = shap.maskers.Independent(
    df[features].astype("float64"),
    max_samples=2000
)

explainer = shap.Explainer(
    model.predict,
    masker
)

print("===== SHAP EXPLAINER READY =====")

# Creates the student database on first run; safe to repeat (see db.init_db)
db.init_db()

print("===== STUDENT DATABASE READY =====")

# The masker keeps mutable buffers (_masked_data, _last_mask) that every
# explanation resets and then delta-updates. Overlapping requests (Flask
# serves them on separate threads) could start from another request's
# leftover buffer state and return wrong SHAP values, so explanations
# are computed one at a time.
explainer_lock = threading.Lock()


def compute_shap(student):

    with explainer_lock:
        return explainer(student)


# ============================================================
# DATASET STATISTICS
# ============================================================

statistics = {}

for feature in features:

    statistics[feature] = {
        "q25": df[feature].quantile(0.25),
        "median": df[feature].median(),
        "q75": df[feature].quantile(0.75)
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

MODEL_INPUT_RANGES = [
    ("attendance", "Attendance", 52, 100),
    ("test1", "Internal Test 1", 18, 40),
    ("test2", "Internal Test 2", 16, 40),
    ("assignment", "Assignment Score", 4, 10),
    ("study_hours", "Daily Study Hours", 1, 5),
]


def validate_student_inputs(data):

    if data is None or not isinstance(data, dict) or not data:
        raise ValueError("Request body must contain JSON data.")

    required_fields = [
        "attendance",
        "test1",
        "test2",
        "assignment",
        "study_hours"
    ]

    values = {}

    for field in required_fields:

        if field not in data:
            raise ValueError(f"Missing required field: {field}")

        if isinstance(data[field], bool):
            raise ValueError("All student inputs must be numeric.")

        try:
            values[field] = float(data[field])
        except (TypeError, ValueError):
            raise ValueError("All student inputs must be numeric.")

        if not math.isfinite(values[field]):
            raise ValueError("All student inputs must be numeric.")

    attendance = values["attendance"]
    test1 = values["test1"]
    test2 = values["test2"]
    assignment = values["assignment"]
    study_hours = values["study_hours"]

    for field, label, low, high in MODEL_INPUT_RANGES:
        if not low <= values[field] <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")

    return attendance, test1, test2, assignment, study_hours


# ============================================================
# EARLY-WARNING RISK CLASSIFICATION
# ============================================================

# Project-defined early-warning boundaries (not official university
# pass/fail thresholds). Single source of truth for all endpoints.
HIGH_CONCERN_BELOW = 40
BORDERLINE_UP_TO = 44

EARLY_WARNING_DISCLAIMER = (
    "This early-warning classification is based on the model's predicted "
    "final-exam performance. It is intended to support timely academic "
    "intervention and does not represent an official university pass/fail "
    "determination."
)


def classify_early_warning(predicted_marks):

    # Classify the unclipped, unrounded model prediction
    if predicted_marks < HIGH_CONCERN_BELOW:
        risk_code = "high_concern"
        risk_level = "High Concern"
        risk_basis = f"Predicted final-exam marks below {HIGH_CONCERN_BELOW}"

    elif predicted_marks <= BORDERLINE_UP_TO:
        risk_code = "borderline"
        risk_level = "Borderline"
        risk_basis = (
            f"Predicted final-exam marks between "
            f"{HIGH_CONCERN_BELOW} and {BORDERLINE_UP_TO}"
        )

    else:
        risk_code = "lower_concern"
        risk_level = "Lower Concern"
        risk_basis = f"Predicted final-exam marks above {BORDERLINE_UP_TO}"

    return {
        "risk_level": risk_level,
        "risk_code": risk_code,
        "risk_basis": risk_basis,
        "risk_disclaimer": EARLY_WARNING_DISCLAIMER
    }


# ============================================================
# PERFORMANCE LEVEL FUNCTION
# ============================================================

def get_performance_level(value, stats):

    if value < stats["q25"]:
        return "Below typical range"

    elif value >= stats["q75"]:
        return "Strong range"

    elif value < stats["median"]:
        return "Slightly below average"

    else:
        return "Typical range"


# ============================================================
# ACTIONABLE RECOMMENDATION KNOWLEDGE BASE
# ============================================================

# Predefined academic guidance per feature and priority. These are
# SHAP-guided suggestions, not claims that changing a feature causes
# a change in marks.
actions = {

    "Attendance (%)": {
        "HIGH":
            "Focus on maintaining regular attendance and identify any barriers that may be affecting your participation in classes.",
        "MEDIUM":
            "Try to maintain more consistent attendance and participation in classes.",
        "POSITIVE":
            "Your attendance is currently contributing positively to the prediction. Continue maintaining regular attendance and participation."
    },

    "Internal Test 1 (out of 40)": {
        "HIGH":
            "Review the topics and mistakes from Internal Test 1, then use targeted practice to strengthen those areas before the final examination.",
        "MEDIUM":
            "Review weaker areas identified from Internal Test 1 and complete additional practice before the final examination.",
        "POSITIVE":
            "Your Internal Test 1 performance is currently contributing positively to the prediction. Maintain the preparation approach that supported this performance."
    },

    "Internal Test 2 (out of 40)": {
        "HIGH":
            "Review the topics and mistakes from Internal Test 2, then use targeted practice to strengthen those areas before the final examination.",
        "MEDIUM":
            "Review weaker areas identified from Internal Test 2 and complete additional practice before the final examination.",
        "POSITIVE":
            "Your Internal Test 2 performance is currently contributing positively to the prediction. Maintain the preparation approach that supported this performance."
    },

    "Assignment Score (out of 10)": {
        "HIGH":
            "Review assignment feedback and errors, complete any outstanding work where possible, and focus on improving understanding of the related topics.",
        "MEDIUM":
            "Improve assignment completion and quality, and review feedback or errors to strengthen understanding.",
        "POSITIVE":
            "Your assignment performance is currently contributing positively to the prediction. Continue maintaining consistent assignment completion and quality."
    },

    "Daily Study Hours": {
        "HIGH":
            "Establish a structured daily study schedule with focused revision and practice, giving additional attention to the factors identified as negative contributors by the model.",
        "MEDIUM":
            "Increase and structure your study time consistently around revision and practice of priority topics.",
        "POSITIVE":
            "Your reported study routine is currently contributing positively to the prediction. Maintain a consistent study routine while continuing to focus on priority topics."
    }
}


# ============================================================
# SHAP-GUIDED RECOMMENDATION RULES
# ============================================================

# Methodology:
# - A SHAP contribution (in marks) is the model's attribution of the
#   prediction to a feature. It identifies model attribution, not causation.
# - Risk level (classify_early_warning) represents overall early-warning
#   urgency and is kept separate: it never changes these thresholds.
# - Feature-specific actions are predefined academic guidance (actions).
#
# Project-defined contribution thresholds (not universal standards):
#   SHAP < -1.0          -> HIGH
#   -1.0 <= SHAP < -0.5  -> MEDIUM
#   -0.5 <= SHAP <= 0.5  -> no recommendation
#   SHAP > 0.5           -> POSITIVE (maintenance guidance)

def classify_shap_priority(shap_value):

    if shap_value < -1:
        return "HIGH"

    elif shap_value < -0.5:
        return "MEDIUM"

    elif shap_value > 0.5:
        return "POSITIVE"

    return None


# Internal tests have already been completed, so their negative-contributor
# messages treat the result as evidence and point to final-exam preparation
# instead of suggesting the test score itself can be changed.
completed_test_messages = {

    "Internal Test 1 (out of 40)": {
        "HIGH":
            "Your Internal Test 1 performance is a strong negative contributor to this prediction. Use the result to identify weak areas and focus your preparation on those areas before the final examination.",
        "MEDIUM":
            "Your Internal Test 1 performance is a negative contributor to this prediction. Review the result to identify weaker areas and focus additional preparation on them before the final examination."
    },

    "Internal Test 2 (out of 40)": {
        "HIGH":
            "Your Internal Test 2 performance is a strong negative contributor to this prediction. Use the result to identify weak areas and focus your preparation on those areas before the final examination.",
        "MEDIUM":
            "Your Internal Test 2 performance is a negative contributor to this prediction. Review the result to identify weaker areas and focus additional preparation on them before the final examination."
    }
}


def build_recommendations(values, contributions):

    # values and contributions are keyed by feature name
    # Only the most negative contribution is described as the strongest
    strongest_negative_feature = min(
        contributions,
        key=contributions.get
    )

    entries = []

    for feature in features:

        value = values[feature]

        shap_value = contributions[feature]

        priority = classify_shap_priority(shap_value)

        if priority is None:
            continue

        performance_level = get_performance_level(
            value,
            statistics[feature]
        )

        if priority in completed_test_messages.get(feature, {}):

            message = completed_test_messages[feature][priority]

        elif priority == "HIGH":

            contributor_wording = (
                "your strongest negative contributor"
                if feature == strongest_negative_feature
                else "a significant negative contributor"
            )

            message = (
                f"High priority: Focus on {feature}. "
                f"Your current value is {value:.1f}, "
                f"and this feature is currently "
                f"{contributor_wording} to the "
                f"predicted score."
            )

        elif priority == "MEDIUM":

            message = (
                f"Consider improving {feature}. "
                f"Your current value is {value:.1f}, "
                f"and it is currently having a moderate "
                f"negative influence on your predicted score."
            )

        else:

            message = (
                f"Maintain your performance in {feature}. "
                f"Your current value ({value:.1f}) is "
                f"contributing positively to your predicted score."
            )

        entries.append((shap_value, {

            "feature": feature,

            "priority": priority,

            "current_value":
                round(float(value), 2),

            "performance_level":
                performance_level,

            "shap_contribution":
                round(float(shap_value), 4),

            "recommendation":
                message,

            "action":
                actions[feature][priority]
        }))

    # Negative contributors first, then positive (maintenance) guidance;
    # within each group, largest unrounded |SHAP| first
    entries.sort(key=lambda entry: (entry[0] > 0, -abs(entry[0])))

    return [recommendation for _, recommendation in entries]


# ============================================================
# HOME API
# ============================================================

@app.route("/")
def home():

    return {
        "status": "success",
        "message": "Student Performance Prediction API is running",
        "model_loaded": True,
        "shap_loaded": True
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        # ----------------------------------------------------
        # Get student inputs
        # ----------------------------------------------------

        attendance, test1, test2, assignment, study_hours = validate_student_inputs(data)

        # ----------------------------------------------------
        # Create student dataframe
        # ----------------------------------------------------

        student = pd.DataFrame([{
            "Attendance (%)": attendance,
            "Internal Test 1 (out of 40)": test1,
            "Internal Test 2 (out of 40)": test2,
            "Assignment Score (out of 10)": assignment,
            "Daily Study Hours": study_hours
        }])

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(student)[0]

        return jsonify({
            "status": "success",
            "predicted_marks": round(float(prediction), 2),
            **classify_early_warning(float(prediction))
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# SHAP EXPLANATION API
# ============================================================

@app.route("/explain", methods=["POST"])
def explain():

    try:

        data = request.get_json(silent=True)

        # ----------------------------------------------------
        # Get student inputs
        # ----------------------------------------------------

        attendance, test1, test2, assignment, study_hours = validate_student_inputs(data)

        # ----------------------------------------------------
        # Create student dataframe
        # ----------------------------------------------------

        student = pd.DataFrame([{
            "Attendance (%)": attendance,
            "Internal Test 1 (out of 40)": test1,
            "Internal Test 2 (out of 40)": test2,
            "Assignment Score (out of 10)": assignment,
            "Daily Study Hours": study_hours
        }])

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(student)[0]

        # ----------------------------------------------------
        # SHAP calculation
        # ----------------------------------------------------

        shap_values = compute_shap(student)

        contributions = dict(
            zip(
                features,
                shap_values.values[0]
            )
        )

        # ----------------------------------------------------
        # Base value
        # ----------------------------------------------------

        base_value = float(
            shap_values.base_values[0]
        )

        return jsonify({

            "status": "success",

            "predicted_marks":
                round(float(prediction), 2),

            **classify_early_warning(float(prediction)),

            "base_value":
                round(base_value, 4),

            "shap_contributions": {

                feature:
                    round(float(value), 4)

                for feature, value
                in contributions.items()
            }
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ============================================================
# RECOMMENDATION API
# ============================================================

@app.route("/recommend", methods=["POST"])
def recommend():

    try:

        data = request.get_json(silent=True)

        # ----------------------------------------------------
        # Get student inputs
        # ----------------------------------------------------

        attendance, test1, test2, assignment, study_hours = validate_student_inputs(data)

        # ----------------------------------------------------
        # Create student dataframe
        # ----------------------------------------------------

        student = pd.DataFrame([{
            "Attendance (%)": attendance,
            "Internal Test 1 (out of 40)": test1,
            "Internal Test 2 (out of 40)": test2,
            "Assignment Score (out of 10)": assignment,
            "Daily Study Hours": study_hours
        }])

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(student)[0]

        # ----------------------------------------------------
        # SHAP calculation
        # ----------------------------------------------------

        shap_values = compute_shap(student)

        contributions = dict(
            zip(
                features,
                shap_values.values[0]
            )
        )

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        recommendations = build_recommendations(
            {feature: student[feature].iloc[0] for feature in features},
            contributions
        )

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return jsonify({

            "status": "success",

            "predicted_marks":
                round(float(prediction), 2),

            **classify_early_warning(float(prediction)),

            "recommendations":
                recommendations
        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 400


# ============================================================
# STUDENT RECORDS: VALIDATION
# ============================================================

# What a real mark can be. Deliberately wider than the model's training
# ranges (validate_student_inputs): a true mark is always stored, and the
# model ranges are checked only when a prediction is requested.
ACADEMIC_RANGES = {
    "attendance": ("Attendance", 0, 100),
    "test1": ("Internal Test 1", 0, 40),
    "test2": ("Internal Test 2", 0, 40),
    "assignment": ("Assignment Score", 0, 10),
    "study_hours": ("Daily Study Hours", 0, 24)
}

FACULTY_FIELDS = ["attendance", "test1", "test2", "assignment"]

RECORD_FIELDS = FACULTY_FIELDS + ["study_hours"]


def validate_academic_values(data, fields):

    if data is None or not isinstance(data, dict) or not data:
        raise ValueError("Request body must contain JSON data.")

    values = {}

    for field in fields:

        label, low, high = ACADEMIC_RANGES[field]

        if field not in data:
            raise ValueError(f"Missing required field: {field}")

        # bool is a subclass of int, so True would otherwise be stored as 1
        if isinstance(data[field], bool):
            raise ValueError(f"{label} must be numeric.")

        try:
            value = float(data[field])
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{label} must be numeric.")

        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")

        values[field] = value

    return values


# ============================================================
# FACULTY BULK CSV UPLOAD
# ============================================================

def normalized_person_name(value):

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def add_bulk_row_error(record, field, reason):

    record["errors"].append({"field": field, "reason": reason})


def read_bulk_csv():

    uploaded_files = request.files
    files = uploaded_files.getlist("file")
    if set(uploaded_files.keys()) != {"file"} or len(files) != 1 or not files[0].filename:
        for _, uploaded_file in uploaded_files.items(multi=True):
            uploaded_file.close()
        raise BulkUploadError("Select one CSV file to upload.")

    uploaded_file = files[0]

    try:
        if not uploaded_file.filename.casefold().endswith(".csv"):
            raise BulkUploadError("Only CSV files are supported.")

        try:
            content = uploaded_file.stream.read(MAX_BULK_UPLOAD_BYTES + 1)
        except OSError as error:
            raise BulkUploadError("The uploaded CSV file could not be read.") from error

        return parse_csv(content)
    finally:
        uploaded_file.close()


def validate_bulk_csv_records(records):

    input_fields = ["attendance", "test1", "test2", "assignment", "study_hours"]

    for record in records:
        if len(record["student_id"]) > 80:
            add_bulk_row_error(record, "student_id", "Student ID must be at most 80 characters.")

        if not record["student_name"]:
            add_bulk_row_error(record, "student_name", "Student Name is required.")
        elif len(record["student_name"]) > 200:
            add_bulk_row_error(record, "student_name", "Student Name must be at most 200 characters.")

        numeric_values = {}
        for field in input_fields:
            raw_value = record["inputs"].get(field, "")
            label = next(label for key, label, _, _ in MODEL_INPUT_RANGES if key == field)

            if not raw_value:
                add_bulk_row_error(record, field, f"{label} is required.")
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                add_bulk_row_error(record, field, f"{label} must be numeric.")
                continue

            if not math.isfinite(value):
                add_bulk_row_error(record, field, f"{label} must be a finite number.")
                continue

            numeric_values[field] = value

        if len(numeric_values) == len(input_fields):
            try:
                validate_student_inputs(numeric_values)
            except ValueError as error:
                for field, label, low, high in MODEL_INPUT_RANGES:
                    if not low <= numeric_values[field] <= high:
                        add_bulk_row_error(
                            record,
                            field,
                            f"{label} must be between {low} and {high}."
                        )
                if not record["errors"]:
                    add_bulk_row_error(record, "record", str(error))
            else:
                record["inputs"] = numeric_values

    eligible_ids = [
        record["student_id"]
        for record in records
        if not record["errors"] and record["student_id"] and record["student_name"]
    ]

    candidates_by_id = db.inspect_bulk_students(eligible_ids)

    for record in records:
        if record["errors"] or not record["student_id"] or not record["student_name"]:
            continue

        candidates = candidates_by_id.get(record["student_id"].casefold(), [])
        if not candidates:
            add_bulk_row_error(
                record,
                "student_id",
                "Student ID does not match an existing student record."
            )
            continue
        if len(candidates) != 1:
            add_bulk_row_error(record, "student_id", "Student ID is ambiguous in the student database.")
            continue

        candidate = candidates[0]
        if not candidate["has_student_account"]:
            add_bulk_row_error(
                record,
                "student_id",
                "The matching student does not have an existing student account."
            )
        elif normalized_person_name(candidate["name"]) != normalized_person_name(record["student_name"]):
            add_bulk_row_error(
                record,
                "student_name",
                "Student Name does not match the existing student identity."
            )
        elif (
            any(value is not None for value in candidate["record_values"])
            or any(value is not None for value in candidate["record_metadata"])
        ):
            add_bulk_row_error(
                record,
                "student_id",
                "An academic record already exists; existing records are not overwritten."
            )
        elif candidate["has_predictions"] or candidate["has_interventions"]:
            add_bulk_row_error(
                record,
                "student_id",
                "Student prediction or intervention history already exists."
            )

    return records


def bulk_validation_payload(records):

    errors = [
        {
            "row_number": record["row_number"],
            "student_id": record["student_id"],
            "student_name": record["student_name"],
            **error
        }
        for record in records
        for error in record["errors"]
    ]
    valid_rows = sum(not record["errors"] for record in records)

    return {
        "status": "success",
        "total_rows": len(records),
        "successful_imports": 0,
        "valid_rows": valid_rows,
        "rejected_rows": len(records) - valid_rows,
        "errors": errors,
        "rows": [
            {
                "row_number": record["row_number"],
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "inputs": record["inputs"],
                "status": "valid" if not record["errors"] else "invalid",
                "errors": record["errors"]
            }
            for record in records
        ]
    }


def bulk_file_error(error):

    errors = error.errors or [{
        "row_number": 0,
        "field": "file",
        "reason": str(error)
    }]

    return jsonify({
        "status": "error",
        "message": str(error),
        "total_rows": 0,
        "successful_imports": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "errors": errors,
        "rows": []
    }), 400


def read_student_registration_csv():

    uploaded_files = request.files
    files = uploaded_files.getlist("file")
    if set(uploaded_files.keys()) != {"file"} or len(files) != 1 or not files[0].filename:
        for _, uploaded_file in uploaded_files.items(multi=True):
            uploaded_file.close()
        raise StudentRegistrationFileError("Select one CSV file to upload.")

    uploaded_file = files[0]
    try:
        if not uploaded_file.filename.casefold().endswith(".csv"):
            raise StudentRegistrationFileError("Only CSV files are supported.")
        try:
            content = uploaded_file.stream.read(MAX_REGISTRATION_BYTES + 1)
        except OSError as error:
            raise StudentRegistrationFileError(
                "The uploaded CSV file could not be read."
            ) from error
        return parse_registration_csv(content)
    finally:
        uploaded_file.close()


def validate_student_registration_records(records):

    eligible = [record for record in records if not record["errors"]]
    conflicts = db.inspect_student_account_conflicts(eligible)
    for record in eligible:
        for error in conflicts.get(record["row_number"], []):
            record["errors"].append(error)
    return records


def student_registration_payload(records):

    errors = [
        {
            "row_number": record["row_number"],
            "student_id": record["student_id"],
            "student_name": record["student_name"],
            "username": record["username"],
            **error,
        }
        for record in records
        for error in record["errors"]
    ]
    valid_rows = sum(not record["errors"] for record in records)
    return {
        "status": "success",
        "total_rows": len(records),
        "valid_rows": valid_rows,
        "rejected_rows": len(records) - valid_rows,
        "successful_registrations": 0,
        "errors": errors,
        "rows": [
            {
                "row_number": record["row_number"],
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "username": record["username"],
                "status": "valid" if not record["errors"] else "invalid",
                "errors": record["errors"],
            }
            for record in records
        ],
    }


def student_registration_file_error(error):

    errors = error.errors or [{
        "row_number": 0,
        "field": "file",
        "reason": str(error),
    }]
    return jsonify({
        "status": "error",
        "message": str(error),
        "total_rows": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "successful_registrations": 0,
        "errors": errors,
        "rows": [],
    }), 400


@app.route("/students/accounts", methods=["POST"])
@require_auth("faculty")
def create_student_account():

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Request body must contain JSON data.", 400)
    try:
        name, roll_no, username, password = db.validate_student_account_values(
            data.get("name"),
            data.get("roll_no"),
            data.get("username"),
            data.get("password"),
        )
        student = db.create_student_account(name, roll_no, username, password)
    except ValueError as error:
        return error_response(str(error), 400)
    except db.StudentAccountConflict as error:
        return error_response(str(error), 409)
    except Exception:
        app.logger.exception("Student account creation failed.")
        return error_response("The student account could not be created.", 500)

    return jsonify({"status": "success", "student": student}), 201


@app.route("/students/registration/validate", methods=["POST"])
@require_auth("faculty")
def validate_student_registration():

    try:
        records = validate_student_registration_records(
            read_student_registration_csv()
        )
    except StudentRegistrationFileError as error:
        return student_registration_file_error(error)
    except RequestEntityTooLarge:
        raise
    except Exception:
        app.logger.exception("Student registration CSV validation failed.")
        return error_response(
            "The registration file could not be validated. Please try again.",
            500,
        )

    return jsonify(student_registration_payload(records))


@app.route("/students/registration/import", methods=["POST"])
@require_auth("faculty")
def import_student_registration():

    try:
        records = validate_student_registration_records(
            read_student_registration_csv()
        )
    except StudentRegistrationFileError as error:
        return student_registration_file_error(error)
    except RequestEntityTooLarge:
        raise
    except Exception:
        app.logger.exception("Student registration CSV validation failed before import.")
        return error_response(
            "The registration file could not be validated. No accounts were created.",
            500,
        )

    payload = student_registration_payload(records)
    valid_records = [record for record in records if not record["errors"]]
    if not valid_records:
        payload["rejected_rows"] = len(records)
        return jsonify(payload)

    try:
        created = db.create_student_accounts(valid_records)
    except db.StudentAccountConflict as error:
        for record in valid_records:
            if record["row_number"] == error.row_number:
                record["errors"].append({
                    "field": error.field,
                    "reason": str(error),
                })
            else:
                record["errors"].append({
                    "field": "file",
                    "reason": (
                        "The batch was rolled back because another account "
                        "conflicted; no accounts were created."
                    ),
                })
        failed_payload = student_registration_payload(records)
        return jsonify({
            **failed_payload,
            "status": "error",
            "message": "Registration conflicted with an account created concurrently. No accounts were created.",
        }), 409
    except Exception:
        app.logger.exception("Student registration transaction failed.")
        for record in valid_records:
            record["errors"].append({
                "field": "file",
                "reason": "The batch failed and was rolled back; no accounts were created.",
            })
        failed_payload = student_registration_payload(records)
        return jsonify({
            **failed_payload,
            "status": "error",
            "message": "Student registration failed. No accounts were created.",
        }), 500

    payload.update({
        "successful_registrations": len(created),
        "results": created,
    })
    return jsonify(payload)


@app.route("/students/bulk-upload/validate", methods=["POST"])
@require_auth("faculty")
def validate_bulk_student_upload():

    try:
        records = validate_bulk_csv_records(read_bulk_csv())
    except BulkUploadError as error:
        return bulk_file_error(error)
    except RequestEntityTooLarge:
        raise
    except Exception:
        app.logger.exception("Bulk CSV validation failed.")
        return jsonify({
            "status": "error",
            "message": "The upload could not be validated. Please try again."
        }), 500

    return jsonify(bulk_validation_payload(records))


@app.route("/students/bulk-upload/import", methods=["POST"])
@require_auth("faculty")
def import_bulk_students():

    try:
        records = validate_bulk_csv_records(read_bulk_csv())
    except BulkUploadError as error:
        return bulk_file_error(error)
    except RequestEntityTooLarge:
        raise
    except Exception:
        app.logger.exception("Bulk CSV validation failed before import.")
        return jsonify({
            "status": "error",
            "message": "The upload could not be validated. No records were imported."
        }), 500

    validation = bulk_validation_payload(records)
    valid_records = [record for record in records if not record["errors"]]
    if not valid_records:
        return jsonify({
            "status": "success",
            "total_rows": len(records),
            "successful_imports": 0,
            "rejected_rows": len(records),
            "errors": validation["errors"],
            "results": []
        })

    prepared_records = []
    failed_record = None
    try:
        for record in valid_records:
            failed_record = record
            prepared_records.append({
                **record,
                "analysis": analyze_inputs(record["inputs"])
            })
    except Exception:
        app.logger.exception("Bulk student analysis failed before the database transaction.")
        analysis_errors = validation["errors"] + [
            {
                "row_number": record["row_number"],
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "field": "prediction",
                "reason": (
                    "Prediction analysis failed for this row; the entire batch was rolled back."
                    if record is failed_record
                    else "The batch was rolled back because a row's prediction analysis failed."
                )
            }
            for record in valid_records
        ]
        return jsonify({
            "status": "error",
            "message": "Prediction analysis failed. No records were imported.",
            "total_rows": len(records),
            "successful_imports": 0,
            "rejected_rows": len(records),
            "errors": analysis_errors,
            "results": []
        }), 500

    try:
        imported = db.commit_bulk_import(prepared_records, g.user["id"])
    except db.BulkImportConflict:
        race_errors = validation["errors"] + [
            {
                "row_number": record["row_number"],
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "field": "student_id",
                "reason": "Student data changed after validation; review the CSV and retry."
            }
            for record in valid_records
        ]
        return jsonify({
            "status": "error",
            "message": "The batch changed after validation. No records were imported.",
            "total_rows": len(records),
            "successful_imports": 0,
            "rejected_rows": len(records),
            "errors": race_errors,
            "results": []
        }), 409
    except Exception:
        app.logger.exception("Bulk student database transaction failed and was rolled back.")
        database_errors = validation["errors"] + [
            {
                "row_number": record["row_number"],
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "field": "database",
                "reason": "The database transaction failed; the entire batch was rolled back."
            }
            for record in valid_records
        ]
        return jsonify({
            "status": "error",
            "message": "The database could not safely save this batch. No records were imported.",
            "total_rows": len(records),
            "successful_imports": 0,
            "rejected_rows": len(records),
            "errors": database_errors,
            "results": []
        }), 500

    results = []
    for item in imported:
        prediction = item["prediction"]
        results.append({
            "row_number": item["row_number"],
            "student_id": item["student_id"],
            "student_name": item["student_name"],
            "prediction_id": prediction["id"],
            "predicted_marks": round(prediction["predicted_marks_raw"], 2),
            "risk_code": prediction["risk_code"],
            "risk_level": prediction["risk_level"],
            "risk_basis": prediction["risk_basis"],
            "risk_disclaimer": EARLY_WARNING_DISCLAIMER,
            "base_value": prediction["base_value"],
            "shap_contributions": prediction["shap_contributions"],
            "recommendations": prediction["recommendations"]
        })

    return jsonify({
        "status": "success",
        "total_rows": len(records),
        "successful_imports": len(results),
        "rejected_rows": len(records) - len(results),
        "errors": validation["errors"],
        "results": results
    })


# ============================================================
# STUDENT RECORDS: ANALYSIS
# ============================================================

def analyze_inputs(inputs):

    # Same model, SHAP explainer, early-warning bands and recommendation rules
    # as /predict, /explain and /recommend, but SHAP is computed once so the
    # stored explanation and recommendations come from the same calculation.
    student = pd.DataFrame([{
        "Attendance (%)": inputs["attendance"],
        "Internal Test 1 (out of 40)": inputs["test1"],
        "Internal Test 2 (out of 40)": inputs["test2"],
        "Assignment Score (out of 10)": inputs["assignment"],
        "Daily Study Hours": inputs["study_hours"]
    }])

    prediction = float(model.predict(student)[0])

    shap_values = compute_shap(student)

    contributions = dict(
        zip(
            features,
            shap_values.values[0]
        )
    )

    recommendations = build_recommendations(
        {feature: student[feature].iloc[0] for feature in features},
        contributions
    )

    early_warning = classify_early_warning(prediction)

    return {
        # Unrounded, as classified; rounded only in API responses
        "predicted_marks": prediction,
        "risk_code": early_warning["risk_code"],
        "risk_level": early_warning["risk_level"],
        "risk_basis": early_warning["risk_basis"],
        "base_value": round(float(shap_values.base_values[0]), 4),
        "shap_contributions": {
            feature: round(float(value), 4)
            for feature, value in contributions.items()
        },
        "recommendations": recommendations
    }


def prediction_response(prediction):

    # A stored prediction in the same field names as the existing endpoints
    return {
        **prediction,
        "predicted_marks": round(prediction["predicted_marks_raw"], 2),
        "risk_disclaimer": EARLY_WARNING_DISCLAIMER
    }


# ============================================================
# STUDENT RECORDS: PREDICTION STATUS
# ============================================================

def record_inputs(record):

    return {field: record[field] for field in RECORD_FIELDS}


def matches_record(prediction, record):

    # A stored prediction is current only while its input snapshot equals
    # the student's current academic record; the snapshot is never changed
    return prediction["inputs"] == record_inputs(record)


def prediction_status(record, latest):

    # Whether a prediction can be generated from the stored record, and
    # whether the latest stored prediction still matches that record
    inputs = record_inputs(record)

    is_current = None if latest is None else matches_record(latest, record)

    if any(inputs[field] is None for field in FACULTY_FIELDS):
        status = "incomplete_record"
        message = "Faculty have not entered all academic marks yet."

    elif inputs["study_hours"] is None:
        status = "awaiting_study_hours"
        message = "Waiting for the student to enter Daily Study Hours."

    else:

        try:
            validate_student_inputs(inputs)

        except ValueError as error:
            return {
                "status": "outside_model_range",
                "can_predict": False,
                "message": (
                    "This record is outside the range the model was trained "
                    f"on, so no prediction can be generated. {error}"
                ),
                "is_latest_prediction_current": is_current
            }

        if latest is None:
            status = "not_predicted"
            message = "No prediction has been generated for this record yet."

        elif not is_current:
            status = "outdated"
            message = "The academic record has changed since the latest prediction."

        else:
            status = "current"
            message = "The latest prediction matches the current academic record."

    return {
        "status": status,
        "can_predict": status in ("not_predicted", "outdated", "current"),
        "message": message,
        "is_latest_prediction_current": is_current
    }


def latest_summary(latest):

    if latest is None:
        return None

    return {
        "id": latest["id"],
        "created_at": latest["created_at"],
        "predicted_marks": round(latest["predicted_marks_raw"], 2),
        "risk_code": latest["risk_code"],
        "risk_level": latest["risk_level"]
    }


def error_response(message, status_code):

    return jsonify({
        "status": "error",
        "message": message
    }), status_code


def student_not_found(student_id):

    return error_response(f"Student {student_id} was not found.", 404)


# ============================================================
# STUDENT RECORDS API
# ============================================================

# Every route requires a signed-in user (Bearer token, see auth.py).
# Permissions are decided here from the token's user, never from the client:
# faculty may read every student, edit the four marks and generate
# predictions; a student may read only their own record and history, edit
# only their own Daily Study Hours and generate only their own predictions.
# For a student, access is checked before existence, so a 403 never reveals
# whether another student id exists.

def student_detail_response(student_id):

    student = db.get_student(student_id)

    if student is None:
        return student_not_found(student_id)

    history = db.list_predictions(student_id)

    latest = history[0] if history else None

    return jsonify({
        "status": "success",
        "student": student,
        "prediction_status": prediction_status(student["record"], latest),
        "latest_prediction": prediction_response(latest) if latest else None
    })


@app.route("/students", methods=["GET"])
@require_auth("faculty")
def students_list():

    latest_by_student = db.latest_predictions()

    students = []

    for student in db.list_students():

        latest = latest_by_student.get(student["id"])

        students.append({
            **student,
            "prediction_status": prediction_status(student["record"], latest),
            "latest_prediction": latest_summary(latest)
        })

    return jsonify({
        "status": "success",
        "students": students
    })


@app.route("/students/<int:student_id>/account-status", methods=["PUT"])
@require_auth("faculty")
def update_student_account_status(student_id):

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("is_active"), bool):
        return error_response(
            "Request body must include a boolean is_active field.",
            400
        )

    if db.get_student(student_id) is None:
        return student_not_found(student_id)

    result = db.set_student_account_active(student_id, data["is_active"])
    if result is None:
        return error_response("Student account not found.", 404)

    return jsonify({"status": "success", **result})


@app.route("/students/<int:student_id>", methods=["GET"])
@require_auth("faculty", "student")
def student_detail(student_id):

    if not can_access_student(g.user, student_id):
        return forbidden()

    return student_detail_response(student_id)


@app.route("/students/<int:student_id>/academic-record", methods=["PUT"])
@require_auth("faculty")
def update_academic_record(student_id):

    # Faculty-provided fields only
    if db.get_student(student_id) is None:
        return student_not_found(student_id)

    try:
        values = validate_academic_values(
            request.get_json(silent=True),
            FACULTY_FIELDS
        )

    except ValueError as error:
        return error_response(str(error), 400)

    db.update_faculty_marks(student_id, **values, updated_by=g.user["id"])

    return student_detail_response(student_id)


@app.route("/students/<int:student_id>/study-hours", methods=["PUT"])
@require_auth("student")
def update_student_study_hours(student_id):

    # Student-provided field only, and only for the signed-in student
    if not can_access_student(g.user, student_id):
        return forbidden()

    if db.get_student(student_id) is None:
        return student_not_found(student_id)

    try:
        values = validate_academic_values(
            request.get_json(silent=True),
            ["study_hours"]
        )

    except ValueError as error:
        return error_response(str(error), 400)

    db.update_study_hours(
        student_id,
        values["study_hours"],
        updated_by=g.user["id"]
    )

    return student_detail_response(student_id)


@app.route("/students/<int:student_id>/predictions", methods=["POST"])
@require_auth("faculty", "student")
def generate_student_prediction(student_id):

    if not can_access_student(g.user, student_id):
        return forbidden()

    # Uses only the stored record: nothing in the request body is a model input
    student = db.get_student(student_id)

    if student is None:
        return student_not_found(student_id)

    history = db.list_predictions(student_id)

    status = prediction_status(
        student["record"],
        history[0] if history else None
    )

    if not status["can_predict"]:
        return error_response(status["message"], 409)

    inputs = record_inputs(student["record"])

    try:
        result = analyze_inputs(inputs)

    except Exception as error:
        return error_response(str(error), 500)

    prediction = db.insert_prediction(
        student_id,
        inputs,
        result,
        created_by_user_id=g.user["id"],
        created_by_role=g.user["role"]
    )

    return jsonify({
        "status": "success",
        "prediction": prediction_response(prediction)
    }), 201


@app.route("/students/<int:student_id>/predictions", methods=["GET"])
@require_auth("faculty", "student")
def student_prediction_history(student_id):

    if not can_access_student(g.user, student_id):
        return forbidden()

    student = db.get_student(student_id)

    if student is None:
        return student_not_found(student_id)

    # Every stored prediction, newest first, exactly as generated; is_current
    # compares its snapshot with the record as it is now
    return jsonify({
        "status": "success",
        "predictions": [
            {
                **prediction_response(prediction),
                "is_current": matches_record(prediction, student["record"])
            }
            for prediction in db.list_predictions(student_id)
        ]
    })


# ============================================================
# INTERVENTIONS (IMPROVEMENT TASKS)
# ============================================================

# Faculty assign improvement tasks to a student, optionally starting from a
# recommendation stored with one of that student's predictions. The task keeps
# a reference to that prediction; the prediction itself is never changed.
# Only the owning student can mark a task completed, and nobody can edit the
# task's text through these routes.

INTERVENTION_CATEGORIES = ["Attendance", "Internal Tests", "Assignments", "Study", "General"]

INTERVENTION_TITLE_MAX = 120

INTERVENTION_DESCRIPTION_MAX = 1000


def required_text(data, field, label, max_length):

    value = data.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")

    if len(value.strip()) > max_length:
        raise ValueError(f"{label} must be at most {max_length} characters.")

    return value.strip()


def validate_intervention(data, student_id):

    if data is None or not isinstance(data, dict) or not data:
        raise ValueError("Request body must contain JSON data.")

    title = required_text(data, "title", "Title", INTERVENTION_TITLE_MAX)

    description = required_text(
        data, "description", "Description", INTERVENTION_DESCRIPTION_MAX
    )

    category = data.get("category")

    if category not in INTERVENTION_CATEGORIES:
        raise ValueError(
            "Category must be one of: " + ", ".join(INTERVENTION_CATEGORIES) + "."
        )

    due_date = data.get("due_date") or None

    if due_date is not None:

        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(due_date)):
                raise ValueError
            date.fromisoformat(due_date)
        except ValueError:
            raise ValueError("Due date must be a valid date (YYYY-MM-DD).")

    prediction_id = data.get("prediction_id")

    feature = data.get("recommendation_feature") or None

    if prediction_id is not None:

        # The referenced prediction must be one of this student's own
        if isinstance(prediction_id, bool) or not isinstance(prediction_id, int):
            raise ValueError("prediction_id must be a prediction id.")

        prediction = db.get_prediction(prediction_id)

        if prediction is None or prediction["student_id"] != student_id:
            raise ValueError("The referenced prediction does not belong to this student.")

        # And a recommendation, if named, must be stored with that prediction
        if feature is not None and feature not in [
            recommendation["feature"]
            for recommendation in prediction["recommendations"]
        ]:
            raise ValueError("The referenced recommendation is not stored with this prediction.")

    elif feature is not None:
        raise ValueError("A recommendation can only be referenced together with its prediction.")

    return {
        "title": title,
        "description": description,
        "category": category,
        "due_date": due_date,
        "prediction_id": prediction_id,
        "recommendation_feature": feature
    }


@app.route("/students/<int:student_id>/interventions", methods=["POST"])
@require_auth("faculty")
def create_intervention(student_id):

    if db.get_student(student_id) is None:
        return student_not_found(student_id)

    try:
        values = validate_intervention(request.get_json(silent=True), student_id)

    except ValueError as error:
        return error_response(str(error), 400)

    intervention = db.insert_intervention(student_id, values, g.user["id"])

    return jsonify({
        "status": "success",
        "intervention": intervention
    }), 201


@app.route("/students/<int:student_id>/interventions", methods=["GET"])
@require_auth("faculty", "student")
def student_interventions(student_id):

    if not can_access_student(g.user, student_id):
        return forbidden()

    if db.get_student(student_id) is None:
        return student_not_found(student_id)

    return jsonify({
        "status": "success",
        "interventions": db.list_interventions(student_id)
    })


@app.route("/interventions", methods=["GET"])
@require_auth("faculty")
def all_interventions():

    # Faculty overview: every student's interventions in one read-only request
    return jsonify({
        "status": "success",
        "interventions": db.list_all_interventions()
    })


@app.route("/interventions/<int:intervention_id>/complete", methods=["PUT"])
@require_auth("student")
def complete_student_intervention(intervention_id):

    intervention = db.get_intervention(intervention_id)

    # A student may complete only their own tasks; a missing id gets the same
    # answer, so other students' task ids cannot be discovered
    if intervention is None or intervention["student_id"] != g.user["student_id"]:
        return error_response("You do not have permission to complete this task.", 403)

    completed = db.complete_intervention(intervention_id, g.user["id"])

    intervention = db.get_intervention(intervention_id)

    if not completed:
        return jsonify({
            "status": "error",
            "message": "This task is already completed.",
            "intervention": intervention
        }), 409

    return jsonify({
        "status": "success",
        "intervention": intervention
    })


# ============================================================
# RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )