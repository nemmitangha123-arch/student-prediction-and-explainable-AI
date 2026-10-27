import csv
import io
import re


MAX_BULK_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_BULK_UPLOAD_ROWS = 500

REQUIRED_HEADERS = {
    "student_id": {
        "studentid", "rollno", "rollnumber", "studentrollno",
        "studentrollnumber", "id"
    },
    "student_name": {"studentname", "fullname", "name"},
    "attendance": {
        "attendance", "attendancepercentage", "attendancepercent",
        "attendancepct"
    },
    "test1": {"test1", "internaltest1", "test1outof40", "internaltest1outof40"},
    "test2": {"test2", "internaltest2", "test2outof40", "internaltest2outof40"},
    "assignment": {
        "assignment", "assignmentscore", "assignmentscoreoutof10"
    },
    "study_hours": {
        "studyhours", "dailystudyhours", "dailystudyhoursperday"
    },
}


class BulkUploadError(ValueError):

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


def normalize_header(value):

    return re.sub(r"[^a-z0-9]", "", value.strip().casefold())


def parse_csv(content):

    if not content:
        raise BulkUploadError("The uploaded CSV file is empty.")

    if len(content) > MAX_BULK_UPLOAD_BYTES:
        raise BulkUploadError("The CSV file exceeds the 2 MB upload limit.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise BulkUploadError("The CSV file must use UTF-8 encoding.") from error

    if "\x00" in text:
        raise BulkUploadError("The CSV file contains invalid null characters.")

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)

    try:
        headers = next(reader, None)
    except csv.Error as error:
        raise BulkUploadError(
            "The CSV header is malformed.",
            [{"row_number": 1, "field": "header", "reason": "Malformed CSV header."}]
        ) from error

    if not headers or not any(header.strip() for header in headers):
        raise BulkUploadError("The CSV file must contain a header row.")

    header_positions = {}
    header_errors = []

    for index, header in enumerate(headers):
        normalized = normalize_header(header)

        for field, aliases in REQUIRED_HEADERS.items():
            if normalized in aliases:
                if field in header_positions:
                    header_errors.append({
                        "row_number": 1,
                        "field": field,
                        "reason": "The header appears more than once."
                    })
                else:
                    header_positions[field] = index
                break

    for field in REQUIRED_HEADERS:
        if field not in header_positions:
            header_errors.append({
                "row_number": 1,
                "field": field,
                "reason": "A required column is missing."
            })

    if header_errors:
        raise BulkUploadError(
            "The CSV must contain each required column exactly once.",
            header_errors
        )

    records = []
    previous_line_number = reader.line_num

    try:
        for values in reader:
            row_number = previous_line_number + 1
            previous_line_number = reader.line_num

            if not values:
                continue

            if len(records) >= MAX_BULK_UPLOAD_ROWS:
                raise BulkUploadError(
                    f"The CSV may contain at most {MAX_BULK_UPLOAD_ROWS} records."
                )

            errors = []
            if len(values) != len(headers):
                errors.append({
                    "field": "row",
                    "reason": (
                        f"Expected {len(headers)} columns, found {len(values)}."
                    )
                })

            record = {
                "row_number": row_number,
                "student_id": "",
                "student_name": "",
                "inputs": {},
                "errors": errors
            }

            for field, index in header_positions.items():
                value = values[index].strip() if index < len(values) else ""
                if field == "student_id":
                    record["student_id"] = value
                elif field == "student_name":
                    record["student_name"] = value
                else:
                    record["inputs"][field] = value

            records.append(record)

    except csv.Error as error:
        malformed_line = max(reader.line_num, 1)
        raise BulkUploadError(
            f"The CSV is malformed near row {malformed_line}.",
            [{
                "row_number": malformed_line,
                "field": "file",
                "reason": "Malformed CSV record."
            }]
        ) from error

    if not records:
        raise BulkUploadError("The CSV contains no student records.")

    first_record_by_id = {}

    for record in records:
        student_id = record["student_id"]
        if not student_id:
            record["errors"].append({
                "field": "student_id",
                "reason": "Student ID is required."
            })
            continue

        normalized_id = student_id.casefold()
        first_record = first_record_by_id.get(normalized_id)

        if first_record is None:
            first_record_by_id[normalized_id] = record
        else:
            reason = f"Student ID is duplicated in row {first_record['row_number']}."
            record["errors"].append({"field": "student_id", "reason": reason})
            if not any(error["field"] == "student_id" for error in first_record["errors"]):
                first_record["errors"].append({
                    "field": "student_id",
                    "reason": f"Student ID is duplicated in row {record['row_number']}."
                })

    return records
