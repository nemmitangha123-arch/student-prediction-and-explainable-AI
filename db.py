import hashlib
import json
import os
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import re

from werkzeug.security import generate_password_hash

# ============================================================
# DATABASE LOCATION
# ============================================================

# A single SQLite file next to the training data. STUDENT_DB_PATH can point
# somewhere else (for example a temporary file while testing).
DB_PATH = os.environ.get(
    "STUDENT_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "student_records.db"
    )
)


def utc_now():

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_connection():

    # One short-lived connection per call: Flask serves requests on separate
    # threads and sqlite3 connections must not be shared between threads.
    # Commits on success, rolls back on error, always closes.
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# SCHEMA
# ============================================================

# Academic ranges (what a real mark can be) are enforced here. The narrower
# ranges the model was trained on are NOT: a record outside them can still be
# stored, it just cannot be used for a prediction (checked in app.py).
SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no     TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    program     TEXT,
    section     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role IN ('faculty', 'student')),
    student_id     INTEGER UNIQUE REFERENCES students (id),
    is_active      INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at     TEXT NOT NULL,
    CHECK (
        (role = 'student' AND student_id IS NOT NULL)
        OR (role = 'faculty' AND student_id IS NULL)
    )
);

-- One current record per student. Faculty provide the four marks;
-- the student provides Daily Study Hours.
CREATE TABLE IF NOT EXISTS academic_records (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id              INTEGER NOT NULL UNIQUE REFERENCES students (id),
    attendance              REAL CHECK (attendance BETWEEN 0 AND 100),
    test1                   REAL CHECK (test1 BETWEEN 0 AND 40),
    test2                   REAL CHECK (test2 BETWEEN 0 AND 40),
    assignment              REAL CHECK (assignment BETWEEN 0 AND 10),
    study_hours             REAL CHECK (study_hours BETWEEN 0 AND 24),
    faculty_updated_by      INTEGER REFERENCES users (id),
    faculty_updated_at      TEXT,
    study_hours_updated_by  INTEGER REFERENCES users (id),
    study_hours_updated_at  TEXT
);

-- Every generated prediction, with a snapshot of the inputs it used and the
-- explanation and recommendations exactly as they were returned at that time.
CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id            INTEGER NOT NULL REFERENCES students (id),
    attendance            REAL NOT NULL,
    test1                 REAL NOT NULL,
    test2                 REAL NOT NULL,
    assignment            REAL NOT NULL,
    study_hours           REAL NOT NULL,
    predicted_marks       REAL NOT NULL,
    risk_code             TEXT NOT NULL,
    risk_level            TEXT NOT NULL,
    risk_basis            TEXT NOT NULL,
    base_value            REAL NOT NULL,
    shap_json             TEXT NOT NULL,
    recommendations_json  TEXT NOT NULL,
    created_by_user_id    INTEGER REFERENCES users (id),
    created_by_role       TEXT CHECK (created_by_role IN ('faculty', 'student')),
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_student
    ON predictions (student_id, id);

-- Improvement tasks faculty assign to a student. prediction_id and
-- recommendation_feature record which stored prediction and recommendation
-- the task came from (both empty for a task written from scratch); the
-- prediction itself is never copied or changed.
CREATE TABLE IF NOT EXISTS interventions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id              INTEGER NOT NULL REFERENCES students (id),
    prediction_id           INTEGER REFERENCES predictions (id),
    recommendation_feature  TEXT,
    title                   TEXT NOT NULL,
    description             TEXT NOT NULL,
    category                TEXT NOT NULL,
    due_date                TEXT,
    status                  TEXT NOT NULL DEFAULT 'assigned'
                            CHECK (status IN ('assigned', 'completed')),
    created_by_user_id      INTEGER NOT NULL REFERENCES users (id),
    created_at              TEXT NOT NULL,
    completed_by_user_id    INTEGER REFERENCES users (id),
    completed_at            TEXT,
    CHECK ((status = 'completed') = (completed_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_interventions_student
    ON interventions (student_id, id);

-- Login sessions. Only a SHA-256 hash of each token is stored, so a copy of
-- the database cannot be used to sign in.
CREATE TABLE IF NOT EXISTS auth_tokens (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash  TEXT NOT NULL UNIQUE,
    user_id     INTEGER NOT NULL REFERENCES users (id),
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
"""


# ============================================================
# SAMPLE STUDENTS
# ============================================================

# Fictional sample students (not real people and not rows of the training
# data). Values are (attendance, test1, test2, assignment, study_hours);
# None means not entered yet. The last three show the states a real record
# can be in: no study hours yet, below the model's attendance range, and a
# newly added student with no marks.
SAMPLE_STUDENTS = [
    ("PBL-2026-001", "Aarav Mehta", (88, 33, 34, 8, 3.5)),
    ("PBL-2026-002", "Diya Kapoor", (76, 28, 27, 7, 2.5)),
    ("PBL-2026-003", "Rohan Verma", (68, 24, 23, 6, 2)),
    ("PBL-2026-004", "Ananya Iyer", (62, 22, 21, 5, 1.5)),
    ("PBL-2026-005", "Meera Nair", (94, 37, 38, 9, 4)),
    ("PBL-2026-006", "Sana Khan", (70, 24, 25, 7, 2)),
    ("PBL-2026-007", "Kabir Singh", (82, 30, 31, 8, None)),
    ("PBL-2026-008", "Arjun Rao", (45, 20, 19, 5, 2)),
    ("PBL-2026-009", "Vikram Joshi", (None, None, None, None, None)),
]

SAMPLE_PROGRAM = "B.Tech CSE"
SAMPLE_SECTION = "A"

# Development accounts for the PBL demonstration. Only Werkzeug hashes of
# these passwords are written to the database. Each sample student signs in
# with their roll number; PBL-2026-001 uses Student@001, and so on.
FACULTY_USERNAME = "faculty"
FACULTY_PASSWORD = "Faculty@2026"


def sample_student_password(roll_no):

    return "Student@" + roll_no[-3:]


def init_db():

    # Safe to run any number of times (Flask's debug reloader imports the app
    # twice): tables are created only if missing, and sample students are
    # inserted only while the students table is empty.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_connection() as connection:

        connection.executescript(SCHEMA)
        _migrate_schema(connection)

        # Take the write lock before checking, so two processes starting at
        # the same moment cannot both seed
        connection.execute("BEGIN IMMEDIATE")

        student_count = connection.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        if student_count == 0:
            _seed_sample_students(connection)

        # Separate check, so a database created before accounts existed
        # (Step 1) also receives them
        user_count = connection.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        if user_count == 0:
            _seed_accounts(connection)


def _migrate_schema(connection):

    user_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(users)")
    }
    if "is_active" not in user_columns:
        connection.execute(
            "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1 "
            "CHECK (is_active IN (0, 1))"
        )


def _seed_accounts(connection):

    now = utc_now()

    connection.execute(
        "INSERT INTO users (username, password_hash, role, student_id, created_at) "
        "VALUES (?, ?, 'faculty', NULL, ?)",
        (FACULTY_USERNAME, generate_password_hash(FACULTY_PASSWORD), now)
    )

    sample_roll_numbers = {roll_no for roll_no, _, _ in SAMPLE_STUDENTS}

    students = connection.execute(
        "SELECT id, roll_no FROM students ORDER BY roll_no"
    ).fetchall()

    for student in students:

        # Accounts only for the known sample students, so no password is
        # ever invented for a student added some other way
        if student["roll_no"] not in sample_roll_numbers:
            continue

        connection.execute(
            "INSERT INTO users (username, password_hash, role, student_id, created_at) "
            "VALUES (?, ?, 'student', ?, ?)",
            (
                student["roll_no"],
                generate_password_hash(sample_student_password(student["roll_no"])),
                student["id"],
                now
            )
        )


def _seed_sample_students(connection):

    now = utc_now()

    for roll_no, name, values in SAMPLE_STUDENTS:

        cursor = connection.execute(
            "INSERT INTO students (roll_no, name, program, section, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (roll_no, name, SAMPLE_PROGRAM, SAMPLE_SECTION, now)
        )

        attendance, test1, test2, assignment, study_hours = values

        has_marks = attendance is not None

        connection.execute(
            "INSERT INTO academic_records ("
            "student_id, attendance, test1, test2, assignment, study_hours, "
            "faculty_updated_at, study_hours_updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cursor.lastrowid,
                attendance, test1, test2, assignment, study_hours,
                now if has_marks else None,
                now if study_hours is not None else None
            )
        )


class StudentAccountConflict(Exception):
    def __init__(self, field, message, row_number=None):
        super().__init__(message)
        self.field = field
        self.row_number = row_number


def validate_student_account_values(name, roll_no, username, password):

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Student name is required.")
    if len(name.strip()) > 120:
        raise ValueError("Student name must be 120 characters or fewer.")

    if not isinstance(roll_no, str) or not roll_no.strip():
        raise ValueError("Student roll number is required.")
    if len(roll_no.strip()) > 80:
        raise ValueError("Student roll number must be 80 characters or fewer.")

    if not isinstance(username, str) or not username.strip():
        raise ValueError("Username is required.")
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,80}", username):
        raise ValueError(
            "Username must be 3 to 80 characters using letters, numbers, dots, "
            "underscores or hyphens."
        )

    if not isinstance(password, str) or not password.strip() or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if len(password) > 256:
        raise ValueError("Password must be 256 characters or fewer.")

    return name.strip(), roll_no.strip(), username, password


def _insert_student_account(connection, name, roll_no, username, password_hash):

    now = utc_now()
    cursor = connection.execute(
        "INSERT INTO students (roll_no, name, program, section, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (roll_no, name, SAMPLE_PROGRAM, SAMPLE_SECTION, now)
    )
    student_id = cursor.lastrowid
    connection.execute(
        "INSERT INTO users (username, password_hash, role, student_id, created_at) "
        "VALUES (?, ?, 'student', ?, ?)",
        (username, password_hash, student_id, now)
    )
    return {
        "id": student_id,
        "roll_no": roll_no,
        "name": name,
        "username": username,
    }


def _check_student_account_conflicts(connection, roll_no, username, row_number=None):

    if connection.execute(
        "SELECT 1 FROM students WHERE roll_no = ? COLLATE NOCASE",
        (roll_no,)
    ).fetchone():
        raise StudentAccountConflict(
            "roll_no", "Student roll number already exists.", row_number
        )

    if connection.execute(
        "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
        (username,)
    ).fetchone():
        raise StudentAccountConflict(
            "username", "Username already exists.", row_number
        )


def create_student_account(name, roll_no, username, password):

    name, roll_no, username, password = validate_student_account_values(
        name, roll_no, username, password
    )
    password_hash = generate_password_hash(password)

    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _check_student_account_conflicts(connection, roll_no, username)
        return _insert_student_account(
            connection, name, roll_no, username, password_hash
        )


def create_student_accounts(records):

    prepared = []
    for record in records:
        name, roll_no, username, password = validate_student_account_values(
            record["student_name"],
            record["student_id"],
            record["username"],
            record["password"],
        )
        prepared.append({
            **record,
            "student_name": name,
            "student_id": roll_no,
            "username": username,
            "password_hash": generate_password_hash(password),
        })

    created = []
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")

        # Check file duplicates and existing identities while holding the
        # write lock, so concurrent registrations cannot partially succeed.
        seen_roll_numbers = set()
        seen_usernames = set()
        for record in prepared:
            roll_key = record["student_id"].casefold()
            username_key = record["username"].casefold()
            if roll_key in seen_roll_numbers:
                raise StudentAccountConflict(
                    "student_id",
                    "Student roll number is duplicated in the uploaded file.",
                    record["row_number"],
                )
            if username_key in seen_usernames:
                raise StudentAccountConflict(
                    "username",
                    "Username is duplicated in the uploaded file.",
                    record["row_number"],
                )
            seen_roll_numbers.add(roll_key)
            seen_usernames.add(username_key)
            _check_student_account_conflicts(
                connection,
                record["student_id"],
                record["username"],
                record["row_number"],
            )

        for record in prepared:
            created.append(_insert_student_account(
                connection,
                record["student_name"],
                record["student_id"],
                record["username"],
                record["password_hash"],
            ))

    return created


# ============================================================
# STUDENTS AND ACADEMIC RECORDS
# ============================================================

STUDENT_COLUMNS = """
    s.id, s.roll_no, s.name, s.program, s.section, s.created_at,
    r.attendance, r.test1, r.test2, r.assignment, r.study_hours,
    r.faculty_updated_by, r.faculty_updated_at,
    r.study_hours_updated_by, r.study_hours_updated_at
"""


def _student_from_row(row):

    return {
        "id": row["id"],
        "roll_no": row["roll_no"],
        "name": row["name"],
        "program": row["program"],
        "section": row["section"],
        "created_at": row["created_at"],
        "account_active": (
            bool(row["account_active"])
            if row["account_active"] is not None
            else None
        ),
        "record": {
            "attendance": row["attendance"],
            "test1": row["test1"],
            "test2": row["test2"],
            "assignment": row["assignment"],
            "study_hours": row["study_hours"],
            "faculty_updated_by": row["faculty_updated_by"],
            "faculty_updated_at": row["faculty_updated_at"],
            "study_hours_updated_by": row["study_hours_updated_by"],
            "study_hours_updated_at": row["study_hours_updated_at"]
        }
    }


def list_students():

    with get_connection() as connection:

        rows = connection.execute(
            f"SELECT {STUDENT_COLUMNS}, u.is_active AS account_active "
            "FROM students s "
            "LEFT JOIN academic_records r ON r.student_id = s.id "
            "LEFT JOIN users u ON u.student_id = s.id AND u.role = 'student' "
            "ORDER BY s.roll_no"
        ).fetchall()

    return [_student_from_row(row) for row in rows]


def get_student(student_id):

    with get_connection() as connection:

        row = connection.execute(
            f"SELECT {STUDENT_COLUMNS}, u.is_active AS account_active "
            "FROM students s "
            "LEFT JOIN academic_records r ON r.student_id = s.id "
            "LEFT JOIN users u ON u.student_id = s.id AND u.role = 'student' "
            "WHERE s.id = ?",
            (student_id,)
        ).fetchone()

    return _student_from_row(row) if row else None


def set_student_account_active(student_id, is_active):

    if not isinstance(is_active, bool):
        raise ValueError("Account status must be a boolean.")

    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        account = connection.execute(
            "SELECT u.id FROM users u "
            "JOIN students s ON s.id = u.student_id "
            "WHERE s.id = ? AND u.role = 'student'",
            (student_id,)
        ).fetchone()
        if account is None:
            return None

        connection.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (int(is_active), account["id"])
        )
        if not is_active:
            connection.execute(
                "DELETE FROM auth_tokens WHERE user_id = ?",
                (account["id"],)
            )

    return {"student_id": student_id, "account_active": is_active}


def inspect_bulk_students(student_ids):

    normalized_ids = {student_id.strip().casefold() for student_id in student_ids if student_id.strip()}
    if not normalized_ids:
        return {}

    placeholders = ", ".join("?" for _ in normalized_ids)

    with get_connection() as connection:

        rows = connection.execute(
            "SELECT s.id, s.roll_no, s.name, "
            "u.id AS student_user_id, "
            "r.attendance, r.test1, r.test2, r.assignment, r.study_hours, "
            "r.faculty_updated_by, r.faculty_updated_at, "
            "r.study_hours_updated_by, r.study_hours_updated_at, "
            "EXISTS (SELECT 1 FROM predictions p WHERE p.student_id = s.id) AS has_predictions, "
            "EXISTS (SELECT 1 FROM interventions i WHERE i.student_id = s.id) AS has_interventions "
            "FROM students s "
            "LEFT JOIN users u ON u.student_id = s.id AND u.role = 'student' "
            "LEFT JOIN academic_records r ON r.student_id = s.id "
            f"WHERE lower(s.roll_no) IN ({placeholders})",
            tuple(normalized_ids)
        ).fetchall()

    by_roll_no = {}
    for row in rows:
        key = row["roll_no"].casefold()
        by_roll_no.setdefault(key, []).append({
            "id": row["id"],
            "roll_no": row["roll_no"],
            "name": row["name"],
            "has_student_account": row["student_user_id"] is not None,
            "record_values": (
                row["attendance"],
                row["test1"],
                row["test2"],
                row["assignment"],
                row["study_hours"]
            ),
            "record_metadata": (
                row["faculty_updated_by"],
                row["faculty_updated_at"],
                row["study_hours_updated_by"],
                row["study_hours_updated_at"]
            ),
            "has_predictions": bool(row["has_predictions"]),
            "has_interventions": bool(row["has_interventions"])
        })

    return by_roll_no


def inspect_student_account_conflicts(records):

    conflicts = {}
    with get_connection() as connection:
        for record in records:
            row_errors = []
            if connection.execute(
                "SELECT 1 FROM students WHERE roll_no = ? COLLATE NOCASE",
                (record["student_id"],)
            ).fetchone():
                row_errors.append({
                    "field": "student_id",
                    "reason": "Student roll number already exists."
                })
            if connection.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
                (record["username"],)
            ).fetchone():
                row_errors.append({
                    "field": "username",
                    "reason": "Username already exists."
                })
            if row_errors:
                conflicts[record["row_number"]] = row_errors

    return conflicts


def update_faculty_marks(student_id, attendance, test1, test2, assignment, updated_by=None):

    # Writes only the four faculty-provided fields; Daily Study Hours is untouched
    now = utc_now()

    with get_connection() as connection:

        connection.execute(
            "INSERT INTO academic_records ("
            "student_id, attendance, test1, test2, assignment, "
            "faculty_updated_by, faculty_updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (student_id) DO UPDATE SET "
            "attendance = excluded.attendance, "
            "test1 = excluded.test1, "
            "test2 = excluded.test2, "
            "assignment = excluded.assignment, "
            "faculty_updated_by = excluded.faculty_updated_by, "
            "faculty_updated_at = excluded.faculty_updated_at",
            (student_id, attendance, test1, test2, assignment, updated_by, now)
        )


def update_study_hours(student_id, study_hours, updated_by=None):

    # Writes only the student-provided field; the faculty marks are untouched
    now = utc_now()

    with get_connection() as connection:

        connection.execute(
            "INSERT INTO academic_records ("
            "student_id, study_hours, study_hours_updated_by, study_hours_updated_at"
            ") VALUES (?, ?, ?, ?) "
            "ON CONFLICT (student_id) DO UPDATE SET "
            "study_hours = excluded.study_hours, "
            "study_hours_updated_by = excluded.study_hours_updated_by, "
            "study_hours_updated_at = excluded.study_hours_updated_at",
            (student_id, study_hours, updated_by, now)
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

def _prediction_from_row(row):

    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "created_at": row["created_at"],
        "created_by_user_id": row["created_by_user_id"],
        "created_by_role": row["created_by_role"],
        "inputs": {
            "attendance": row["attendance"],
            "test1": row["test1"],
            "test2": row["test2"],
            "assignment": row["assignment"],
            "study_hours": row["study_hours"]
        },
        # Stored unrounded: the risk band was assigned from this exact value
        "predicted_marks_raw": row["predicted_marks"],
        "risk_code": row["risk_code"],
        "risk_level": row["risk_level"],
        "risk_basis": row["risk_basis"],
        "base_value": row["base_value"],
        "shap_contributions": json.loads(row["shap_json"]),
        "recommendations": json.loads(row["recommendations_json"])
    }


def insert_prediction(student_id, inputs, result, created_by_user_id=None, created_by_role=None):

    with get_connection() as connection:
        row = _insert_prediction(
            connection,
            student_id,
            inputs,
            result,
            created_by_user_id,
            created_by_role
        )

    return _prediction_from_row(row)


def _insert_prediction(
    connection,
    student_id,
    inputs,
    result,
    created_by_user_id=None,
    created_by_role=None
):

    cursor = connection.execute(
        "INSERT INTO predictions ("
        "student_id, attendance, test1, test2, assignment, study_hours, "
        "predicted_marks, risk_code, risk_level, risk_basis, base_value, "
        "shap_json, recommendations_json, "
        "created_by_user_id, created_by_role, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            student_id,
            inputs["attendance"],
            inputs["test1"],
            inputs["test2"],
            inputs["assignment"],
            inputs["study_hours"],
            result["predicted_marks"],
            result["risk_code"],
            result["risk_level"],
            result["risk_basis"],
            result["base_value"],
            json.dumps(result["shap_contributions"]),
            json.dumps(result["recommendations"]),
            created_by_user_id,
            created_by_role,
            utc_now()
        )
    )

    return connection.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (cursor.lastrowid,)
    ).fetchone()


class BulkImportConflict(Exception):
    pass


def _normalized_name(value):

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def commit_bulk_import(records, created_by_user_id):

    candidates_to_import = []

    with get_connection() as connection:

        connection.execute("BEGIN IMMEDIATE")

        # Recheck all identities and records while holding the write lock.
        # This prevents a stale preview from overwriting a concurrent update.
        for record in records:
            candidates = connection.execute(
                "SELECT s.id, s.roll_no, s.name, u.id AS student_user_id, "
                "r.id AS academic_record_id, "
                "r.attendance, r.test1, r.test2, r.assignment, r.study_hours, "
                "r.faculty_updated_by, r.faculty_updated_at, "
                "r.study_hours_updated_by, r.study_hours_updated_at, "
                "EXISTS (SELECT 1 FROM predictions p WHERE p.student_id = s.id) AS has_predictions, "
                "EXISTS (SELECT 1 FROM interventions i WHERE i.student_id = s.id) AS has_interventions "
                "FROM students s "
                "LEFT JOIN users u ON u.student_id = s.id AND u.role = 'student' "
                "LEFT JOIN academic_records r ON r.student_id = s.id "
                "WHERE lower(s.roll_no) = lower(?)",
                (record["student_id"],)
            ).fetchall()

            if len(candidates) != 1:
                raise BulkImportConflict("A student identity changed after validation.")

            candidate = candidates[0]
            if candidate["student_user_id"] is None:
                raise BulkImportConflict("A student account changed after validation.")
            if _normalized_name(candidate["name"]) != _normalized_name(record["student_name"]):
                raise BulkImportConflict("A student name changed after validation.")
            if any(candidate[field] is not None for field in (
                "attendance", "test1", "test2", "assignment", "study_hours"
            )) or any(candidate[field] is not None for field in (
                "faculty_updated_by",
                "faculty_updated_at",
                "study_hours_updated_by",
                "study_hours_updated_at"
            )):
                raise BulkImportConflict("An academic record changed after validation.")
            if candidate["has_predictions"] or candidate["has_interventions"]:
                raise BulkImportConflict("Student history changed after validation.")

            candidates_to_import.append((
                record,
                candidate["id"],
                candidate["academic_record_id"]
            ))

        now = utc_now()
        imported = []

        for record, student_id, record_id in candidates_to_import:
            inputs = record["inputs"]
            if record_id is None:
                connection.execute(
                    "INSERT INTO academic_records ("
                    "student_id, attendance, test1, test2, assignment, study_hours, "
                    "faculty_updated_by, faculty_updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        student_id,
                        inputs["attendance"],
                        inputs["test1"],
                        inputs["test2"],
                        inputs["assignment"],
                        inputs["study_hours"],
                        created_by_user_id,
                        now
                    )
                )
            else:
                updated = connection.execute(
                    "UPDATE academic_records SET "
                    "attendance = ?, test1 = ?, test2 = ?, assignment = ?, study_hours = ?, "
                    "faculty_updated_by = ?, faculty_updated_at = ? "
                    "WHERE student_id = ? "
                    "AND attendance IS NULL AND test1 IS NULL AND test2 IS NULL "
                    "AND assignment IS NULL AND study_hours IS NULL",
                    (
                        inputs["attendance"],
                        inputs["test1"],
                        inputs["test2"],
                        inputs["assignment"],
                        inputs["study_hours"],
                        created_by_user_id,
                        now,
                        student_id
                    )
                )
                if updated.rowcount != 1:
                    raise BulkImportConflict("An academic record changed during import.")

            prediction_row = _insert_prediction(
                connection,
                student_id,
                inputs,
                record["analysis"],
                created_by_user_id,
                "faculty"
            )
            imported.append({
                "student_id": record["student_id"],
                "student_name": record["student_name"],
                "row_number": record["row_number"],
                "prediction": _prediction_from_row(prediction_row)
            })

    return imported


def list_predictions(student_id):

    # Newest first
    with get_connection() as connection:

        rows = connection.execute(
            "SELECT * FROM predictions WHERE student_id = ? ORDER BY id DESC",
            (student_id,)
        ).fetchall()

    return [_prediction_from_row(row) for row in rows]


def latest_predictions():

    # The most recent prediction of every student that has one, keyed by student id
    with get_connection() as connection:

        rows = connection.execute(
            "SELECT * FROM predictions WHERE id IN ("
            "SELECT MAX(id) FROM predictions GROUP BY student_id"
            ")"
        ).fetchall()

    return {row["student_id"]: _prediction_from_row(row) for row in rows}


# ============================================================
# USERS AND LOGIN TOKENS
# ============================================================

TOKEN_LIFETIME = timedelta(hours=8)


def hash_token(token):

    # Tokens are long random values, so a fast SHA-256 is enough here
    # (unlike passwords, which use Werkzeug's slow salted hashing)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_user_by_username(username):

    # Includes password_hash for checking a login; never sent to a client.
    with get_connection() as connection:

        row = connection.execute(
            "SELECT id, username, password_hash, role, student_id, is_active FROM users "
            "WHERE username = ? COLLATE NOCASE",
            (username,)
        ).fetchone()

    return dict(row) if row else None


def create_token(user_id, token):

    now = datetime.now(timezone.utc)

    expires_at = (now + TOKEN_LIFETIME).isoformat(timespec="seconds")

    with get_connection() as connection:

        # Housekeeping: expired sessions are no longer useful
        connection.execute(
            "DELETE FROM auth_tokens WHERE expires_at <= ?",
            (now.isoformat(timespec="seconds"),)
        )

        connection.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (hash_token(token), user_id, now.isoformat(timespec="seconds"), expires_at)
        )

    return expires_at


def get_user_by_token(token):

    # The signed-in user for a token that exists and has not expired
    with get_connection() as connection:

        row = connection.execute(
            "SELECT u.id, u.username, u.role, u.student_id, t.expires_at "
            "FROM auth_tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token_hash = ? AND t.expires_at > ? AND u.is_active = 1",
            (hash_token(token), utc_now())
        ).fetchone()

    return dict(row) if row else None


def delete_token(token):

    with get_connection() as connection:

        connection.execute(
            "DELETE FROM auth_tokens WHERE token_hash = ?",
            (hash_token(token),)
        )


# ============================================================
# PREDICTION LOOKUP
# ============================================================

def get_prediction(prediction_id):

    with get_connection() as connection:

        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,)
        ).fetchone()

    return _prediction_from_row(row) if row else None


# ============================================================
# INTERVENTIONS (IMPROVEMENT TASKS)
# ============================================================

# The source prediction's id and date are read through the reference,
# not stored again with the intervention
INTERVENTION_SELECT = """
    SELECT i.*, p.created_at AS prediction_created_at
    FROM interventions i
    LEFT JOIN predictions p ON p.id = i.prediction_id
"""


def _intervention_from_row(row):

    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "prediction_id": row["prediction_id"],
        "prediction_created_at": row["prediction_created_at"],
        "recommendation_feature": row["recommendation_feature"],
        "title": row["title"],
        "description": row["description"],
        "category": row["category"],
        "due_date": row["due_date"],
        "status": row["status"],
        "created_by_user_id": row["created_by_user_id"],
        "created_at": row["created_at"],
        "completed_by_user_id": row["completed_by_user_id"],
        "completed_at": row["completed_at"]
    }


def get_intervention(intervention_id):

    with get_connection() as connection:

        row = connection.execute(
            INTERVENTION_SELECT + " WHERE i.id = ?",
            (intervention_id,)
        ).fetchone()

    return _intervention_from_row(row) if row else None


def list_interventions(student_id):

    # Newest first
    with get_connection() as connection:

        rows = connection.execute(
            INTERVENTION_SELECT + " WHERE i.student_id = ? ORDER BY i.id DESC",
            (student_id,)
        ).fetchall()

    return [_intervention_from_row(row) for row in rows]


def list_all_interventions():

    # Every student's interventions, newest first, with the student's name and
    # roll number (read-only; for the faculty overview pages)
    with get_connection() as connection:

        rows = connection.execute(
            "SELECT i.*, p.created_at AS prediction_created_at, "
            "s.name AS student_name, s.roll_no AS student_roll_no "
            "FROM interventions i "
            "LEFT JOIN predictions p ON p.id = i.prediction_id "
            "JOIN students s ON s.id = i.student_id "
            "ORDER BY i.id DESC"
        ).fetchall()

    return [
        {
            **_intervention_from_row(row),
            "student_name": row["student_name"],
            "student_roll_no": row["student_roll_no"]
        }
        for row in rows
    ]


def insert_intervention(student_id, values, created_by_user_id):

    with get_connection() as connection:

        cursor = connection.execute(
            "INSERT INTO interventions ("
            "student_id, prediction_id, recommendation_feature, title, "
            "description, category, due_date, status, "
            "created_by_user_id, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, 'assigned', ?, ?)",
            (
                student_id,
                values["prediction_id"],
                values["recommendation_feature"],
                values["title"],
                values["description"],
                values["category"],
                values["due_date"],
                created_by_user_id,
                utc_now()
            )
        )

    return get_intervention(cursor.lastrowid)


def complete_intervention(intervention_id, completed_by_user_id):

    # Only an assigned task changes; returns False if it was already completed
    with get_connection() as connection:

        cursor = connection.execute(
            "UPDATE interventions SET status = 'completed', "
            "completed_by_user_id = ?, completed_at = ? "
            "WHERE id = ? AND status = 'assigned'",
            (completed_by_user_id, utc_now(), intervention_id)
        )

    return cursor.rowcount == 1
