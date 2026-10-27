"""Integration tests for faculty-created student accounts and CSV registration."""

import atexit
import csv
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
import uuid
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if "STUDENT_DB_PATH" not in os.environ:
    _test_database_dir = tempfile.TemporaryDirectory(prefix="student-registration-tests-")
    atexit.register(_test_database_dir.cleanup)
    os.environ["STUDENT_DB_PATH"] = os.path.join(
        _test_database_dir.name,
        "student_records.db",
    )

import app  # noqa: E402
import db  # noqa: E402
from werkzeug.security import check_password_hash  # noqa: E402


REGISTRATION_HEADERS = [
    "Student Name",
    "Student Roll Number",
    "Username",
    "Password",
]

ACADEMIC_HEADERS = [
    "Student ID",
    "Student Name",
    "Attendance",
    "Internal Test 1",
    "Internal Test 2",
    "Assignment Score",
    "Daily Study Hours",
]


class StudentRegistrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()
        response = cls.client.post("/auth/login", json={
            "username": db.FACULTY_USERNAME,
            "password": db.FACULTY_PASSWORD,
        })
        cls.faculty_token = response.get_json()["token"]
        cls.faculty_headers = {"Authorization": f"Bearer {cls.faculty_token}"}
        student_login = cls.client.post("/auth/login", json={
            "username": "PBL-2026-001",
            "password": "Student@001",
        })
        cls.student_token = student_login.get_json()["token"]
        cls.student_headers = {"Authorization": f"Bearer {cls.student_token}"}

    @staticmethod
    def make_csv(headers, rows):
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue().encode("utf-8")

    @staticmethod
    def registration_row(suffix=None, **overrides):
        suffix = suffix or uuid.uuid4().hex[:10]
        row = [
            f"Registration Student {suffix}",
            f"REG-{suffix}",
            f"reg.user.{suffix}",
            f"Secret-{suffix}-Password",
        ]
        return overrides.get("row", row)

    def upload(self, endpoint, content, filename="students.csv", headers=None):
        return self.client.post(
            endpoint,
            data={"file": (io.BytesIO(content), filename)},
            headers=self.faculty_headers if headers is None else headers,
            content_type="multipart/form-data",
        )

    def test_individual_creation_links_student_and_login_account(self):
        suffix = uuid.uuid4().hex[:10]
        password = f"Unique-{suffix}-Password"
        response = self.client.post(
            "/students/accounts",
            headers=self.faculty_headers,
            json={
                "name": f"Individual Student {suffix}",
                "roll_no": f"IND-{suffix}",
                "username": f"individual.{suffix}",
                "password": password,
            },
        )
        self.assertEqual(response.status_code, 201)
        student = response.get_json()["student"]
        self.assertEqual(student["roll_no"], f"IND-{suffix}")
        self.assertNotIn("password", student)
        self.assertNotIn(password, response.get_data(as_text=True))

        with db.get_connection() as connection:
            account = connection.execute(
                "SELECT u.username, u.role, u.student_id, u.password_hash, s.roll_no "
                "FROM users u JOIN students s ON s.id = u.student_id "
                "WHERE s.id = ?",
                (student["id"],),
            ).fetchone()
            record = connection.execute(
                "SELECT id FROM academic_records WHERE student_id = ?",
                (student["id"],),
            ).fetchone()
            predictions = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE student_id = ?",
                (student["id"],),
            ).fetchone()[0]
            interventions = connection.execute(
                "SELECT COUNT(*) FROM interventions WHERE student_id = ?",
                (student["id"],),
            ).fetchone()[0]

        self.assertEqual(account["role"], "student")
        self.assertEqual(account["student_id"], student["id"])
        self.assertEqual(account["roll_no"], student["roll_no"])
        self.assertTrue(check_password_hash(account["password_hash"], password))
        self.assertNotEqual(account["password_hash"], password)
        self.assertIsNone(record)
        self.assertEqual(predictions, 0)
        self.assertEqual(interventions, 0)

        login = self.client.post("/auth/login", json={
            "username": student["username"],
            "password": password,
        })
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.get_json()["user"]["role"], "student")

    def test_individual_creation_rejects_duplicate_roll_and_username_case_insensitively(self):
        suffix = uuid.uuid4().hex[:10]
        first = self.client.post(
            "/students/accounts",
            headers=self.faculty_headers,
            json={
                "name": "First Duplicate Test",
                "roll_no": f"DUP-{suffix}",
                "username": f"duplicate.{suffix}",
                "password": "Duplicate-Password",
            },
        )
        self.assertEqual(first.status_code, 201)
        details = [
            (
                {"roll_no": f"dup-{suffix}", "username": f"other.{suffix}"},
                "roll number already exists",
            ),
            (
                {"roll_no": f"OTHER-{suffix}", "username": f"DUPLICATE.{suffix}"},
                "Username already exists",
            ),
        ]
        for changes, expected_message in details:
            body = {
                "name": "Second Duplicate Test",
                "roll_no": f"SECOND-{suffix}",
                "username": f"second.{suffix}",
                "password": "Duplicate-Password",
                **changes,
            }
            with self.subTest(changes=changes):
                response = self.client.post(
                    "/students/accounts",
                    headers=self.faculty_headers,
                    json=body,
                )
                self.assertEqual(response.status_code, 409)
                self.assertIn(expected_message, response.get_json()["message"])

    def test_individual_creation_requires_fields_and_valid_username_password(self):
        suffix = uuid.uuid4().hex[:8]
        incomplete = self.client.post(
            "/students/accounts",
            headers=self.faculty_headers,
            json={"name": "", "roll_no": f"MISS-{suffix}", "username": "", "password": ""},
        )
        self.assertEqual(incomplete.status_code, 400)
        self.assertIn("Student name is required", incomplete.get_json()["message"])

        short_password = self.client.post(
            "/students/accounts",
            headers=self.faculty_headers,
            json={
                "name": "Short Password Student",
                "roll_no": f"SHORT-{suffix}",
                "username": f"short.{suffix}",
                "password": "short",
            },
        )
        self.assertEqual(short_password.status_code, 400)
        self.assertIn("at least 8", short_password.get_json()["message"])

    def test_account_creation_requires_authenticated_faculty(self):
        payload = {
            "name": "Unauthorized Attempt",
            "roll_no": f"UNAUTH-{uuid.uuid4().hex[:8]}",
            "username": f"unauth.{uuid.uuid4().hex[:8]}",
            "password": "Unauthorized-Password",
        }
        missing_auth = self.client.post("/students/accounts", json=payload)
        self.assertEqual(missing_auth.status_code, 401)
        student_auth = self.client.post(
            "/students/accounts",
            headers=self.student_headers,
            json=payload,
        )
        self.assertEqual(student_auth.status_code, 403)

    def test_bulk_registration_creates_multiple_accounts_without_password_disclosure(self):
        suffix_a, suffix_b = uuid.uuid4().hex[:9], uuid.uuid4().hex[:9]
        rows = [
            [f"Bulk Student {suffix_a}", f"CSV-{suffix_a}", f"csv.{suffix_a}", f"Password-{suffix_a}"],
            [f"Bulk Student {suffix_b}", f"CSV-{suffix_b}", f"csv.{suffix_b}", f"Password-{suffix_b}"],
        ]
        content = self.make_csv(REGISTRATION_HEADERS, rows)
        original = db.get_student(1)
        original_record = dict(original["record"])
        original_history_count = len(db.list_predictions(1))
        original_interventions = db.list_interventions(1)
        with db.get_connection() as connection:
            original_account = dict(connection.execute(
                "SELECT username, role, student_id, password_hash "
                "FROM users WHERE student_id = ?",
                (1,),
            ).fetchone())

        preview = self.upload("/students/registration/validate", content)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.get_json()["total_rows"], 2)
        self.assertEqual(preview.get_json()["valid_rows"], 2)
        self.assertNotIn(rows[0][3], preview.get_data(as_text=True))

        imported = self.upload("/students/registration/import", content)
        self.assertEqual(imported.status_code, 200)
        body = imported.get_json()
        self.assertEqual(body["successful_registrations"], 2)
        self.assertEqual(body["rejected_rows"], 0)
        self.assertNotIn(rows[0][3], imported.get_data(as_text=True))
        self.assertNotIn("password_hash", imported.get_data(as_text=True))

        for created, source in zip(body["results"], rows):
            with self.subTest(roll_no=source[1]):
                login = self.client.post("/auth/login", json={
                    "username": source[2],
                    "password": source[3],
                })
                self.assertEqual(login.status_code, 200)
                with db.get_connection() as connection:
                    role = connection.execute(
                        "SELECT role FROM users WHERE username = ?",
                        (source[2],),
                    ).fetchone()["role"]
                    academic = connection.execute(
                        "SELECT id FROM academic_records WHERE student_id = ?",
                        (created["id"],),
                    ).fetchone()
                    predictions = connection.execute(
                        "SELECT COUNT(*) FROM predictions WHERE student_id = ?",
                        (created["id"],),
                    ).fetchone()[0]
                    interventions = connection.execute(
                        "SELECT COUNT(*) FROM interventions WHERE student_id = ?",
                        (created["id"],),
                    ).fetchone()[0]
                self.assertEqual(role, "student")
                self.assertIsNone(academic)
                self.assertEqual(predictions, 0)
                self.assertEqual(interventions, 0)

        unchanged = db.get_student(1)
        self.assertEqual(unchanged["record"], original_record)
        self.assertEqual(len(db.list_predictions(1)), original_history_count)
        self.assertEqual(db.list_interventions(1), original_interventions)
        with db.get_connection() as connection:
            unchanged_account = dict(connection.execute(
                "SELECT username, role, student_id, password_hash "
                "FROM users WHERE student_id = ?",
                (1,),
            ).fetchone())
        self.assertEqual(unchanged_account, original_account)

    def test_bulk_validation_rejects_duplicate_roll_numbers_and_usernames(self):
        suffix = uuid.uuid4().hex[:8]
        rows = [
            ["Duplicate One", f"DUP-{suffix}", f"dup.{suffix}", "Password-One"],
            ["Duplicate Two", f"dup-{suffix}", f"DUP.{suffix}", "Password-Two"],
        ]
        response = self.upload(
            "/students/registration/validate",
            self.make_csv(REGISTRATION_HEADERS, rows),
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["valid_rows"], 0)
        self.assertEqual(payload["rejected_rows"], 2)
        self.assertTrue(all(row["status"] == "invalid" for row in payload["rows"]))
        self.assertNotIn(rows[0][3], response.get_data(as_text=True))
        self.assertNotIn(rows[1][3], response.get_data(as_text=True))

    def test_bulk_validation_reports_empty_fields_and_existing_identities(self):
        suffix = uuid.uuid4().hex[:8]
        rows = [
            ["", f"EMPTY-{suffix}", f"empty.{suffix}", "Password-Empty"],
            ["Already Exists", "PBL-2026-001", f"new.{suffix}", "Password-Existing"],
            ["Username Conflict", f"NEW-{suffix}", "faculty", "Password-Conflict"],
        ]
        response = self.upload(
            "/students/registration/validate",
            self.make_csv(REGISTRATION_HEADERS, rows),
        )
        payload = response.get_json()
        self.assertEqual(payload["valid_rows"], 0)
        reasons = " ".join(error["reason"] for error in payload["errors"])
        self.assertIn("Student Name is required", reasons)
        self.assertIn("Student roll number already exists", reasons)
        self.assertIn("Username already exists", reasons)

    def test_bulk_registration_rejects_empty_malformed_missing_header_and_unsupported_files(self):
        empty = self.upload("/students/registration/validate", b"")
        self.assertEqual(empty.status_code, 400)
        malformed = self.upload(
            "/students/registration/validate",
            b'Student Name,Student Roll Number,Username,Password\n"unfinished',
        )
        self.assertEqual(malformed.status_code, 400)
        missing_header = self.upload(
            "/students/registration/validate",
            self.make_csv(REGISTRATION_HEADERS[:-1], [["Name", "ROLL-1", "user.one"]]),
        )
        self.assertEqual(missing_header.status_code, 400)
        unsupported = self.upload(
            "/students/registration/validate",
            self.make_csv(REGISTRATION_HEADERS, [self.registration_row()]),
            filename="students.xlsx",
        )
        self.assertEqual(unsupported.status_code, 400)
        oversized = self.upload(
            "/students/registration/validate",
            b"x" * (2 * 1024 * 1024 + 1),
        )
        self.assertEqual(oversized.status_code, 400)
        over_limit_rows = [
            [f"Over Limit {index}", f"LIMIT-{index}", f"limit.{index}", "Over-Limit-Password"]
            for index in range(501)
        ]
        too_many_rows = self.upload(
            "/students/registration/validate",
            self.make_csv(REGISTRATION_HEADERS, over_limit_rows),
        )
        self.assertEqual(too_many_rows.status_code, 400)

    def test_registration_requires_authenticated_faculty_for_both_bulk_routes(self):
        content = self.make_csv(REGISTRATION_HEADERS, [self.registration_row()])
        for endpoint in (
            "/students/registration/validate",
            "/students/registration/import",
        ):
            with self.subTest(endpoint=endpoint):
                unauthenticated = self.upload(endpoint, content, headers={})
                self.assertEqual(unauthenticated.status_code, 401)
                student = self.upload(endpoint, content, headers=self.student_headers)
                self.assertEqual(student.status_code, 403)

    def test_failed_bulk_creation_rolls_back_all_new_students_and_accounts(self):
        suffix_a, suffix_b = uuid.uuid4().hex[:9], uuid.uuid4().hex[:9]
        rows = [
            [f"Rollback {suffix_a}", f"ROLLBACK-{suffix_a}", f"rollback.{suffix_a}", f"Password-{suffix_a}"],
            [f"Rollback {suffix_b}", f"ROLLBACK-{suffix_b}", f"rollback.{suffix_b}", f"Password-{suffix_b}"],
        ]
        original_insert = db._insert_student_account
        calls = 0

        def fail_on_second_insert(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated database failure")
            return original_insert(*args, **kwargs)

        with patch.object(db, "_insert_student_account", side_effect=fail_on_second_insert):
            response = self.upload(
                "/students/registration/import",
                self.make_csv(REGISTRATION_HEADERS, rows),
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["successful_registrations"], 0)
        with db.get_connection() as connection:
            for row in rows:
                student = connection.execute(
                    "SELECT id FROM students WHERE roll_no = ?",
                    (row[1],),
                ).fetchone()
                account = connection.execute(
                    "SELECT id FROM users WHERE username = ?",
                    (row[2],),
                ).fetchone()
                self.assertIsNone(student)
                self.assertIsNone(account)

    def test_concurrent_duplicate_creation_only_creates_one_account(self):
        suffix = uuid.uuid4().hex[:10]
        name = f"Concurrent Student {suffix}"
        roll_no = f"CONCURRENT-{suffix}"
        username = f"concurrent.{suffix}"
        password = f"Concurrent-{suffix}-Password"

        def create_account():
            try:
                return db.create_student_account(name, roll_no, username, password)
            except db.StudentAccountConflict as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: create_account(), range(2)))

        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        self.assertEqual(sum(isinstance(result, db.StudentAccountConflict) for result in results), 1)
        with db.get_connection() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM students WHERE roll_no = ?",
                    (roll_no,),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM users WHERE username = ?",
                    (username,),
                ).fetchone()[0],
                1,
            )

    def test_new_registration_is_eligible_for_existing_academic_upload_validation(self):
        suffix = uuid.uuid4().hex[:9]
        row = [f"Academic Eligible {suffix}", f"ACADEMIC-{suffix}", f"academic.{suffix}", f"Password-{suffix}"]
        created = self.upload(
            "/students/registration/import",
            self.make_csv(REGISTRATION_HEADERS, [row]),
        )
        self.assertEqual(created.status_code, 200)
        academic_row = [
            row[1], row[0], 91, 32, 36, 7, 3,
        ]
        validation = self.upload(
            "/students/bulk-upload/validate",
            self.make_csv(ACADEMIC_HEADERS, [academic_row]),
        )
        self.assertEqual(validation.status_code, 200)
        self.assertEqual(validation.get_json()["valid_rows"], 1)
        self.assertEqual(validation.get_json()["rows"][0]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
