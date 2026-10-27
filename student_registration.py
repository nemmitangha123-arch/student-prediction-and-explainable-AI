import csv
import io
import re


MAX_REGISTRATION_BYTES = 2 * 1024 * 1024
MAX_REGISTRATION_ROWS = 500

REQUIRED_HEADERS = {
    "student_name": {"studentname", "fullname", "name"},
    "student_id": {
        "studentid", "studentrollno", "studentrollnumber",
        "rollno", "rollnumber", "id"
    },
    "username": {"username", "loginusername"},
    "password": {"password", "loginpassword"},
}


class StudentRegistrationFileError(ValueError):

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


def normalize_header(value):

    return re.sub(r"[^a-z0-9]", "", value.strip().casefold())


def parse_registration_csv(content):

    if not content:
        raise StudentRegistrationFileError("The uploaded CSV file is empty.")
    if len(content) > MAX_REGISTRATION_BYTES:
        raise StudentRegistrationFileError("The CSV file exceeds the 2 MB upload limit.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise StudentRegistrationFileError(
            "The CSV file must use UTF-8 encoding."
        ) from error

    if "\x00" in text:
        raise StudentRegistrationFileError(
            "The CSV file contains invalid null characters."
        )

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = next(reader, None)
    except csv.Error as error:
        raise StudentRegistrationFileError(
            "The CSV header is malformed.",
            [{"row_number": 1, "field": "header", "reason": "Malformed CSV header."}],
        ) from error

    if not headers or not any(header.strip() for header in headers):
        raise StudentRegistrationFileError("The CSV file must contain a header row.")

    positions = {}
    header_errors = []
    for index, header in enumerate(headers):
        normalized = normalize_header(header)
        for field, aliases in REQUIRED_HEADERS.items():
            if normalized in aliases:
                if field in positions:
                    header_errors.append({
                        "row_number": 1,
                        "field": field,
                        "reason": "The header appears more than once.",
                    })
                else:
                    positions[field] = index
                break

    for field in REQUIRED_HEADERS:
        if field not in positions:
            header_errors.append({
                "row_number": 1,
                "field": field,
                "reason": "A required column is missing.",
            })
    if header_errors:
        raise StudentRegistrationFileError(
            "The CSV must contain each required column exactly once.",
            header_errors,
        )

    records = []
    previous_line_number = reader.line_num
    try:
        for values in reader:
            row_number = previous_line_number + 1
            previous_line_number = reader.line_num
            if not values:
                continue
            if len(records) >= MAX_REGISTRATION_ROWS:
                raise StudentRegistrationFileError(
                    f"The CSV may contain at most {MAX_REGISTRATION_ROWS} records."
                )

            record = {
                "row_number": row_number,
                "student_name": "",
                "student_id": "",
                "username": "",
                "password": "",
                "errors": [],
            }
            if len(values) != len(headers):
                record["errors"].append({
                    "field": "row",
                    "reason": f"Expected {len(headers)} columns, found {len(values)}.",
                })

            for field, index in positions.items():
                value = values[index] if index < len(values) else ""
                if field != "password":
                    value = value.strip()
                record[field] = value
                if not value.strip():
                    record["errors"].append({
                        "field": field,
                        "reason": f"{field.replace('_', ' ').title()} is required.",
                    })

            if record["student_name"] and len(record["student_name"]) > 120:
                record["errors"].append({
                    "field": "student_name",
                    "reason": "Student name must be 120 characters or fewer.",
                })
            if record["student_id"] and len(record["student_id"]) > 80:
                record["errors"].append({
                    "field": "student_id",
                    "reason": "Student roll number must be 80 characters or fewer.",
                })
            if record["username"] and not re.fullmatch(
                r"[A-Za-z0-9._-]{3,80}", record["username"]
            ):
                record["errors"].append({
                    "field": "username",
                    "reason": (
                        "Username must be 3 to 80 characters using letters, "
                        "numbers, dots, underscores or hyphens."
                    ),
                })
            if record["password"]:
                if len(record["password"]) < 8:
                    record["errors"].append({
                        "field": "password",
                        "reason": "Password must be at least 8 characters.",
                    })
                elif len(record["password"]) > 256:
                    record["errors"].append({
                        "field": "password",
                        "reason": "Password must be 256 characters or fewer.",
                    })
            records.append(record)

    except csv.Error as error:
        malformed_line = max(reader.line_num, 1)
        raise StudentRegistrationFileError(
            f"The CSV is malformed near row {malformed_line}.",
            [{
                "row_number": malformed_line,
                "field": "file",
                "reason": "Malformed CSV record.",
            }],
        ) from error

    if not records:
        raise StudentRegistrationFileError("The CSV contains no student records.")

    first_row_by_roll = {}
    first_row_by_username = {}
    for record in records:
        if record["student_id"]:
            key = record["student_id"].casefold()
            first = first_row_by_roll.get(key)
            if first is None:
                first_row_by_roll[key] = record
            else:
                _mark_duplicate(first, record, "student_id", "Student roll number")
        if record["username"]:
            key = record["username"].casefold()
            first = first_row_by_username.get(key)
            if first is None:
                first_row_by_username[key] = record
            else:
                _mark_duplicate(first, record, "username", "Username")

    return records


def _mark_duplicate(first, duplicate, field, label):

    duplicate_reason = f"{label} is duplicated in row {first['row_number']}."
    first_reason = f"{label} is duplicated in row {duplicate['row_number']}."
    duplicate["errors"].append({"field": field, "reason": duplicate_reason})
    if not any(error["field"] == field for error in first["errors"]):
        first["errors"].append({"field": field, "reason": first_reason})
