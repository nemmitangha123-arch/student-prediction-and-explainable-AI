"""Tests for the SHAP-guided recommendation rules in app.py.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import threading
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# app.py loads the model and dataset with relative paths
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import app  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ATTENDANCE = "Attendance (%)"
TEST1 = "Internal Test 1 (out of 40)"
TEST2 = "Internal Test 2 (out of 40)"
ASSIGNMENT = "Assignment Score (out of 10)"
STUDY = "Daily Study Hours"

VALUES = {
    ATTENDANCE: 60,
    TEST1: 20,
    TEST2: 18,
    ASSIGNMENT: 5,
    STUDY: 1.5,
}


def contributions_with(**overrides):
    # All features neutral (no recommendation) unless overridden
    contributions = {feature: 0.0 for feature in app.features}
    contributions.update(overrides)
    return contributions


class ShapThresholdTests(unittest.TestCase):

    def test_threshold_boundaries(self):
        cases = [
            (-1.01, "HIGH"),
            (-1.00, "MEDIUM"),
            (-0.99, "MEDIUM"),
            (-0.51, "MEDIUM"),
            (-0.50, None),
            (-0.49, None),
            (0.50, None),
            (0.51, "POSITIVE"),
        ]
        for shap_value, expected in cases:
            with self.subTest(shap_value=shap_value):
                self.assertEqual(app.classify_shap_priority(shap_value), expected)

    def test_boundaries_through_build_recommendations(self):
        cases = [
            (-1.01, "HIGH"),
            (-1.00, "MEDIUM"),
            (-0.51, "MEDIUM"),
            (-0.50, None),
            (0.50, None),
            (0.51, "POSITIVE"),
        ]
        for shap_value, expected in cases:
            with self.subTest(shap_value=shap_value):
                recommendations = app.build_recommendations(
                    VALUES, contributions_with(**{ATTENDANCE: shap_value})
                )
                if expected is None:
                    self.assertEqual(recommendations, [])
                else:
                    self.assertEqual(len(recommendations), 1)
                    self.assertEqual(recommendations[0]["priority"], expected)


class OrderingTests(unittest.TestCase):

    def test_negative_recommendations_sorted_by_shap_magnitude(self):
        contributions = {
            ATTENDANCE: -9.70,
            TEST1: -10.56,
            TEST2: -13.03,
            ASSIGNMENT: -3.73,
            STUDY: -3.81,
        }
        recommendations = app.build_recommendations(VALUES, contributions)
        self.assertEqual(
            [r["feature"] for r in recommendations],
            [TEST2, TEST1, ATTENDANCE, STUDY, ASSIGNMENT],
        )

    def test_sorting_uses_unrounded_shap_values(self):
        # Both round to -2.0 at 4 decimals only if rounding were applied first
        contributions = contributions_with(**{
            ATTENDANCE: -2.00001,
            TEST1: -2.00004,
        })
        recommendations = app.build_recommendations(VALUES, contributions)
        self.assertEqual([r["feature"] for r in recommendations], [TEST1, ATTENDANCE])

    def test_positive_recommendations_come_after_negative(self):
        contributions = {
            ATTENDANCE: 3.18,
            TEST1: -0.75,
            TEST2: 0.51,
            ASSIGNMENT: -2.40,
            STUDY: 2.38,
        }
        recommendations = app.build_recommendations(VALUES, contributions)
        self.assertEqual(
            [(r["feature"], r["priority"]) for r in recommendations],
            [
                (ASSIGNMENT, "HIGH"),
                (TEST1, "MEDIUM"),
                (ATTENDANCE, "POSITIVE"),
                (STUDY, "POSITIVE"),
                (TEST2, "POSITIVE"),
            ],
        )

    def test_all_qualifying_recommendations_are_returned(self):
        contributions = {
            ATTENDANCE: -9.70,
            TEST1: -10.56,
            TEST2: -13.03,
            ASSIGNMENT: -3.73,
            STUDY: -3.81,
        }
        self.assertEqual(len(app.build_recommendations(VALUES, contributions)), 5)


class ActionTextTests(unittest.TestCase):

    def test_high_action_text(self):
        recommendations = app.build_recommendations(
            VALUES, contributions_with(**{TEST2: -13.03})
        )
        self.assertEqual(
            recommendations[0]["action"],
            "Review the topics and mistakes from Internal Test 2, then use targeted "
            "practice to strengthen those areas before the final examination.",
        )

    def test_medium_action_text(self):
        recommendations = app.build_recommendations(
            VALUES, contributions_with(**{ASSIGNMENT: -0.75})
        )
        self.assertEqual(
            recommendations[0]["action"],
            "Improve assignment completion and quality, and review feedback or "
            "errors to strengthen understanding.",
        )

    def test_positive_action_text(self):
        recommendations = app.build_recommendations(
            VALUES, contributions_with(**{STUDY: 0.51})
        )
        self.assertEqual(
            recommendations[0]["action"],
            "Your reported study routine is currently contributing positively to "
            "the prediction. Maintain a consistent study routine while continuing "
            "to focus on priority topics.",
        )

    def test_every_feature_has_all_three_actions(self):
        for feature in app.features:
            with self.subTest(feature=feature):
                self.assertEqual(
                    set(app.actions[feature]), {"HIGH", "MEDIUM", "POSITIVE"}
                )


class CompletedTestAndStudyWordingTests(unittest.TestCase):

    EXPECTED_TEST_MESSAGES = {
        (TEST1, "HIGH"):
            "Your Internal Test 1 performance is a strong negative contributor to this prediction. "
            "Use the result to identify weak areas and focus your preparation on those areas before the final examination.",
        (TEST1, "MEDIUM"):
            "Your Internal Test 1 performance is a negative contributor to this prediction. "
            "Review the result to identify weaker areas and focus additional preparation on them before the final examination.",
        (TEST2, "HIGH"):
            "Your Internal Test 2 performance is a strong negative contributor to this prediction. "
            "Use the result to identify weak areas and focus your preparation on those areas before the final examination.",
        (TEST2, "MEDIUM"):
            "Your Internal Test 2 performance is a negative contributor to this prediction. "
            "Review the result to identify weaker areas and focus additional preparation on them before the final examination.",
    }

    SHAP_FOR_PRIORITY = {"HIGH": -2.0, "MEDIUM": -0.75, "POSITIVE": 2.0}

    def recommendation_for(self, feature, priority):
        recommendations = app.build_recommendations(
            VALUES, contributions_with(**{feature: self.SHAP_FOR_PRIORITY[priority]})
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["priority"], priority)
        return recommendations[0]

    def test_completed_test_messages_use_finalized_wording(self):
        for (feature, priority), expected in self.EXPECTED_TEST_MESSAGES.items():
            with self.subTest(feature=feature, priority=priority):
                message = self.recommendation_for(feature, priority)["recommendation"]
                self.assertEqual(message, expected)
                self.assertNotIn("improve", message.lower())
                self.assertNotIn(f"Focus on {feature}", message)
                self.assertNotIn("Consider improving", message)

    def test_completed_test_actions_unchanged(self):
        for feature in (TEST1, TEST2):
            for priority in ("HIGH", "MEDIUM", "POSITIVE"):
                with self.subTest(feature=feature, priority=priority):
                    self.assertEqual(
                        self.recommendation_for(feature, priority)["action"],
                        app.actions[feature][priority],
                    )

    def test_other_messages_keep_existing_templates(self):
        self.assertTrue(
            self.recommendation_for(ATTENDANCE, "HIGH")["recommendation"].startswith(
                "High priority: Focus on Attendance (%). Your current value is 60.0"
            )
        )
        self.assertTrue(
            self.recommendation_for(ASSIGNMENT, "MEDIUM")["recommendation"].startswith(
                "Consider improving Assignment Score (out of 10). Your current value is 5.0"
            )
        )
        self.assertTrue(
            self.recommendation_for(TEST1, "POSITIVE")["recommendation"].startswith(
                "Maintain your performance in Internal Test 1 (out of 40)."
            )
        )

    def test_study_hours_actions(self):
        self.assertEqual(
            self.recommendation_for(STUDY, "HIGH")["action"],
            "Establish a structured daily study schedule with focused revision and practice, "
            "giving additional attention to the factors identified as negative contributors by the model.",
        )
        self.assertNotIn("areas identified by the model", app.actions[STUDY]["HIGH"])
        self.assertEqual(
            app.actions[STUDY]["MEDIUM"],
            "Increase and structure your study time consistently around revision and practice of priority topics.",
        )


class RiskClassificationTests(unittest.TestCase):

    def test_risk_boundaries_unchanged(self):
        cases = [
            (39.99, "High Concern", "high_concern"),
            (40.00, "Borderline", "borderline"),
            (44.00, "Borderline", "borderline"),
            (44.01, "Lower Concern", "lower_concern"),
        ]
        for marks, level, code in cases:
            with self.subTest(marks=marks):
                result = app.classify_early_warning(marks)
                self.assertEqual(result["risk_level"], level)
                self.assertEqual(result["risk_code"], code)


class RecommendEndpointTests(unittest.TestCase):

    TOP_LEVEL_FIELDS = {
        "status", "predicted_marks", "risk_level", "risk_code",
        "risk_basis", "risk_disclaimer", "recommendations",
    }
    RECOMMENDATION_FIELDS = {
        "feature", "priority", "current_value", "performance_level",
        "shap_contribution", "recommendation", "action",
    }

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def post(self, payload):
        response = self.client.post("/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_response_fields_present(self):
        body = self.post({"attendance": 91, "test1": 32, "test2": 36, "assignment": 7, "study_hours": 3})
        self.assertEqual(set(body), self.TOP_LEVEL_FIELDS)
        self.assertTrue(body["recommendations"])
        for recommendation in body["recommendations"]:
            self.assertEqual(set(recommendation), self.RECOMMENDATION_FIELDS)

    def test_endpoint_orders_real_shap_values(self):
        body = self.post({"attendance": 60, "test1": 20, "test2": 18, "assignment": 5, "study_hours": 1.5})
        shap_values = [r["shap_contribution"] for r in body["recommendations"]]
        negatives = [v for v in shap_values if v < 0]
        positives = [v for v in shap_values if v > 0]
        self.assertEqual(shap_values, negatives + positives)
        self.assertEqual(negatives, sorted(negatives, key=abs, reverse=True))
        self.assertEqual(positives, sorted(positives, key=abs, reverse=True))

    def test_invalid_input_still_rejected(self):
        response = self.client.post("/recommend", json={"attendance": 10, "test1": 32, "test2": 36, "assignment": 7, "study_hours": 3})
        self.assertEqual(response.status_code, 400)


class ShapConsistencyTests(unittest.TestCase):
    """Regression tests for the shared-explainer race between /explain and /recommend."""

    CASE_A = {"attendance": 91, "test1": 32, "test2": 36, "assignment": 7, "study_hours": 3}
    CASE_B = {"attendance": 60, "test1": 20, "test2": 18, "assignment": 5, "study_hours": 1.5}

    EXPECTED_A = {
        ATTENDANCE: 2.3813,
        TEST1: -0.1007,
        TEST2: 3.1849,
        ASSIGNMENT: -0.7546,
        STUDY: 0.5083,
    }

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    @staticmethod
    def student_frame(payload):
        return pd.DataFrame([{
            ATTENDANCE: payload["attendance"],
            TEST1: payload["test1"],
            TEST2: payload["test2"],
            ASSIGNMENT: payload["assignment"],
            STUDY: payload["study_hours"],
        }])

    def endpoint_shap(self, client, payload):
        explain = client.post("/explain", json=payload).get_json()["shap_contributions"]
        recommend = {
            r["feature"]: r["shap_contribution"]
            for r in client.post("/recommend", json=payload).get_json()["recommendations"]
        }
        return explain, recommend

    def test_feature_order_is_fixed(self):
        self.assertEqual(app.features, [ATTENDANCE, TEST1, TEST2, ASSIGNMENT, STUDY])

    def test_case_a_reference_values(self):
        explain, _ = self.endpoint_shap(self.client, self.CASE_A)
        self.assertEqual(explain, self.EXPECTED_A)

    def test_explain_and_recommend_return_identical_shap(self):
        for name, payload in (("A", self.CASE_A), ("B", self.CASE_B)):
            with self.subTest(case=name):
                explain, recommend = self.endpoint_shap(self.client, payload)
                self.assertEqual(set(explain), set(app.features))
                # /recommend only lists features that pass the thresholds
                for feature, value in recommend.items():
                    self.assertEqual(value, explain[feature], feature)

    def test_concurrent_explanations_are_consistent(self):
        # Overlapping calls on the shared explainer, as with parallel Flask
        # requests. A short switch interval makes thread interleaving frequent;
        # nothing in the explainer or masker is patched.
        reference = {
            name: app.compute_shap(self.student_frame(payload)).values[0].copy()
            for name, payload in (("A", self.CASE_A), ("B", self.CASE_B))
        }
        failures = []
        failures_lock = threading.Lock()

        def worker(name, payload):
            frame = self.student_frame(payload)
            for _ in range(100):
                values = app.compute_shap(frame).values[0]
                if not np.array_equal(values, reference[name]):
                    with failures_lock:
                        failures.append((name, values.tolist()))

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [
                threading.Thread(target=worker, args=(name, payload))
                for name, payload in (("A", self.CASE_A), ("B", self.CASE_B), ("A", self.CASE_A), ("B", self.CASE_B))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous_interval)

        self.assertEqual(failures, [], f"{len(failures)} of 400 concurrent explanations differed")

    def test_concurrent_endpoints_agree(self):
        results = []
        results_lock = threading.Lock()

        def worker(endpoint, payload):
            client = app.app.test_client()
            for _ in range(15):
                body = client.post(endpoint, json=payload).get_json()
                with results_lock:
                    results.append((endpoint, payload["attendance"], body))

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [
                threading.Thread(target=worker, args=(endpoint, payload))
                for endpoint in ("/explain", "/recommend")
                for payload in (self.CASE_A, self.CASE_B)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous_interval)

        reference = {
            payload["attendance"]: self.endpoint_shap(self.client, payload)[0]
            for payload in (self.CASE_A, self.CASE_B)
        }
        for endpoint, attendance, body in results:
            expected = reference[attendance]
            if endpoint == "/explain":
                self.assertEqual(body["shap_contributions"], expected)
            else:
                for r in body["recommendations"]:
                    self.assertEqual(r["shap_contribution"], expected[r["feature"]], r["feature"])


if __name__ == "__main__":
    unittest.main()
