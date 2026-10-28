"""Analytic tests generated in memory; no study observations are included."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from healthy_reconstruction.analysis import (
    NUMERICAL_TIE_TOLERANCE_MM,
    POLICIES,
    ScoreRecord,
    aggregate_regional_takes,
    aggregate_take_scores,
    analyze_file,
    cluster_auc,
    pattern_frequencies,
    regional_summary,
    side_contrasts,
    summarize_contrasts,
    summarize_followup,
    wilson_interval,
    within_visit_variability,
)


class AUCTests(unittest.TestCase):
    @staticmethod
    def records():
        # Distinct repeated-visit weights create different session/subject AUCs.
        values = (("a", "reference", 1, 0), ("b", "reference", 1, 10),
                  ("c", "comparison", 1, 1), ("c", "comparison", 2, 2),
                  ("d", "comparison", 1, 20))
        return [ScoreRecord(p, g, v, "task", (x, 7.0)) for p, g, v, x in values]

    def test_four_estimands_and_ties(self):
        result = cluster_auc(self.records(), ("varying", "tied"), ("task",), n_bootstrap=32, seed=4)
        np.testing.assert_allclose(result.estimate[:, 0, 0], [2 / 3, 3 / 4, 3 / 4, 3 / 4])
        np.testing.assert_allclose(result.estimate[..., 1], 0.5)
        np.testing.assert_allclose(result.estimate[:, 0], result.estimate[:, 1])
        self.assertEqual(result.bootstrap.shape, (32, 4, 2, 2))
        self.assertEqual(result.interval().shape, (2, 4, 2, 2))
        delta = result.paired_difference("varying", "tied")
        np.testing.assert_allclose(delta["estimate"], result.estimate[..., 0] - 0.5)
        np.testing.assert_allclose(delta["interval"], np.quantile(result.bootstrap[..., 0] - 0.5, [.025, .975], axis=0))

    def test_shared_cluster_draws_against_direct_pair_enumeration(self):
        result = cluster_auc(self.records(), ("varying", "tied"), ("task",), n_bootstrap=40, seed=8)
        reference_values = ([0.0], [10.0])
        comparison_values = ([1.0, 2.0], [20.0])

        def pair_auc(x, y):
            return sum(float(b > a) + .5 * float(b == a) for a in x for b in y) / (len(x) * len(y))

        for draw, (hc, cc) in enumerate(zip(result.reference_draw_counts, result.comparison_draw_counts)):
            hi = np.repeat(np.arange(2), hc)
            ci = np.repeat(np.arange(2), cc)
            for policy_index, policy in enumerate(POLICIES):
                if policy == "equal_subject_weighted_sessions":
                    expected = np.mean([pair_auc(reference_values[h], comparison_values[c]) for h in hi for c in ci])
                else:
                    hvalues, cvalues = [], []
                    for h in hi:
                        a = reference_values[h]
                        hvalues.extend([np.median(a)] if policy in POLICIES[1:3] else a)
                    for c in ci:
                        b = comparison_values[c]
                        cvalues.extend([np.median(b)] if policy == "participant_median" else [b[0]] if policy == POLICIES[2] else b)
                    expected = pair_auc(hvalues, cvalues)
                self.assertAlmostEqual(result.bootstrap[draw, policy_index, 0, 0], expected)
        duplicate = cluster_auc(self.records(), ("varying", "tied"), ("task",), n_bootstrap=40, seed=8)
        np.testing.assert_array_equal(result.bootstrap, duplicate.bootstrap)

    def test_joint_rejection_preserves_missing_task_support(self):
        records = self.records()
        records += [ScoreRecord("a", "reference", 1, "other", (0., 7.)),
                    ScoreRecord("c", "comparison", 1, "other", (1., 7.)),
                    ScoreRecord("c", "comparison", 2, "other", (2., 7.)),
                    ScoreRecord("d", "comparison", 1, "other", (20., 7.))]
        result = cluster_auc(records, ("varying", "tied"), ("task", "other"), n_bootstrap=50, seed=2)
        self.assertTrue((result.reference_draw_counts[:, 0] > 0).all())
        self.assertTrue(np.isfinite(result.bootstrap).all())
        np.testing.assert_allclose(result.estimate[:, -1], result.estimate[:, :-1].mean(axis=1))

    def test_invalid_pairing_is_rejected(self):
        rows = self.records()
        with self.assertRaises(ValueError):
            cluster_auc(rows + rows[:1], ("varying", "tied"), ("task",), n_bootstrap=0)
        with self.assertRaises(ValueError):
            cluster_auc(rows, ("varying",), ("task",), n_bootstrap=0)
        with self.assertRaises(ValueError):
            cluster_auc(rows, ("varying", "tied"), ("absent",), n_bootstrap=0)
        np.testing.assert_allclose(aggregate_take_scores(np.array([[0, 8], [2, 3], [10, 4]])), [2, 4])
        with self.assertRaises(ValueError):
            aggregate_take_scores(np.array([[1, np.nan]]))

    def test_private_prediction_loader_aggregates_and_excludes_extra_fields(self):
        rows = self.records()
        records = []
        for index, row in enumerate(rows):
            records.append({"sample_id": f"generated_{index}", "participant_id": row.participant_id,
                            "group": row.group, "visit_index": row.visit_index, "task": row.task,
                            "scores": {"tied": row.scores[1], "ae": row.scores[0]},
                            "extra_field": "must_not_be_forwarded"})
        # A second generated take changes the per-unit median before AUC.
        records.append({**records[0], "sample_id": "another_generated_take", "scores": {"ae": 2., "tied": 7.}})
        # Comparison repetitions retain an AE median of one and an IQR of one;
        # the other method has a different scale and must stay out of follow-up.
        records.append({**records[2], "sample_id": "comparison_repeat_low", "scores": {"ae": 0., "tied": 100.}})
        records.append({**records[2], "sample_id": "comparison_repeat_high", "scores": {"ae": 2., "tied": 200.}})
        payload = {"schema_version": 1, "methods": ["tied", "ae"], "records": records}
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "private_input.json", Path(directory) / "private_report.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            before = source.read_bytes()
            result = analyze_file(source, destination, iterations=16, seed=1,
                                  reference_group="reference", comparison_group="comparison")
            self.assertEqual(before, source.read_bytes())
            self.assertEqual(result, json.loads(destination.read_text(encoding="utf-8")))
            self.assertNotIn("must_not_be_forwarded", destination.read_text(encoding="utf-8"))
            self.assertEqual(result["bootstrap"]["iterations"], 16)
            self.assertNotIn("score_omission",result)
            self.assertEqual(result["ordinal_followup"]["status"], "complete_task_visits")
            self.assertEqual(result["methods"],["tied","ae"])
            self.assertEqual(np.asarray(result["auc_estimate"]).shape,(4,2,2))
            self.assertEqual(result["ordinal_followup"]["methods"],["ae"])
            endpoint=result["ordinal_followup"]["endpoints"][0]
            self.assertEqual(endpoint["global_delta"],[1.])
            self.assertEqual(endpoint["task_delta"],[[1.]])
            self.assertTrue(all(len(r["global_score"])==1 for r in result["ordinal_followup"]["visits"]))
            variability=result["within_visit_take_variability"]
            self.assertTrue(all(r["method"]=="ae" and len(r["median"])==1 for r in variability))
            repeated=next(r for r in variability if r["participant_id"]=="c" and r["visit_index"]==1)
            self.assertEqual(repeated["median"],[1.])
            self.assertEqual(repeated["iqr"],[1.])
            self.assertEqual(result["regional_patterns"]["methods"],["ae"])
            with self.assertRaises(ValueError):
                analyze_file(source, source, iterations=0)
            payload["records"].append(records[0])
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_file(source, destination, iterations=0)

    def test_file_loader_keeps_all_five_cohort_methods_when_followup_is_incomplete(self):
        from healthy_reconstruction.inference import METHODS
        records=[]
        for index,(person,group,visit,task,value) in enumerate([
                ("reference_person","reference",1,"one",0.),
                ("reference_person","reference",1,"two",0.),
                ("comparison_person","comparison",1,"one",1.),
                ("comparison_person","comparison",1,"two",1.),
                ("comparison_person","comparison",2,"one",3.)]):
            records.append({"sample_id":f"generated_{index}","participant_id":person,
                            "group":group,"visit_index":visit,"task":task,
                            "scores":{m:value+i for i,m in enumerate(METHODS)}})
        payload={"schema_version":1,"methods":list(METHODS),"records":records}
        with tempfile.TemporaryDirectory() as directory:
            source,destination=Path(directory)/'input.json',Path(directory)/'report.json'
            source.write_text(json.dumps(payload))
            result=analyze_file(source,destination,iterations=0,
                                reference_group="reference",comparison_group="comparison")
            self.assertEqual(result["methods"],list(METHODS))
            self.assertEqual(np.asarray(result["auc_estimate"]).shape,(4,3,5))
            self.assertEqual(len(result["paired_method_differences"]),10)
            self.assertEqual(result["ordinal_followup"]["status"],"unavailable")
            self.assertEqual(result["ordinal_followup"]["methods"],["ae"])
            self.assertTrue(all(r["method"]=="ae" and len(r["median"])==1
                                for r in result["within_visit_take_variability"]))
            payload["methods"]=[m for m in METHODS if m!="ae"]
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'requires the ae'):
                analyze_file(source,destination,iterations=0)


class FollowupTests(unittest.TestCase):
    def test_task_balanced_endpoints_use_ordinal_visits(self):
        rows = [ScoreRecord("local", "comparison", v, t, (s,)) for v, a, b in [(1, 2., 8.), (3, 4., 4.), (5, 5., 3.)] for t, s in [("one", a), ("two", b)]]
        result = summarize_followup(reversed(rows), ("method",), ("one", "two"))
        np.testing.assert_allclose([x["global_score"][0] for x in result["visits"]], [5, 4, 4])
        endpoint = result["endpoints"][0]
        self.assertEqual(endpoint["n_visits"], 3)
        self.assertEqual(endpoint["first_visit_index"], 1)
        self.assertEqual(endpoint["last_visit_index"], 5)
        np.testing.assert_allclose(endpoint["task_delta"][:, 0], [3, -5])
        np.testing.assert_allclose(endpoint["global_delta"], [-1])
        np.testing.assert_allclose(endpoint["global_delta_percent"], [-20])
        np.testing.assert_array_equal(endpoint["n_tasks_smaller"], [1])
        with self.assertRaises(ValueError):
            summarize_followup(rows[1:], ("method",), ("one", "two"))

    def test_repetition_iqr_is_not_between_visit_repeatability(self):
        values = {("local", 1, "task"): np.array([[0.], [2.], [4.], [6.]]), ("local", 2, "task"): np.array([[3.]])}
        output = within_visit_variability(values)
        self.assertEqual(output[0]["n_takes"], 4)
        np.testing.assert_allclose(output[0]["median"], [3.])
        np.testing.assert_allclose(output[0]["iqr"], [3.])
        self.assertTrue(np.isnan(output[1]["iqr"]).all())


class RegionalPatternTests(unittest.TestCase):
    def test_nested_finite_means_and_full_grid_coverage(self):
        values = np.array([[0., np.nan], [2., np.nan], [4., 8.]])
        before = values.copy()
        result = regional_summary({"counter_direction": values}, [0, 1])["counter_direction"]
        self.assertAlmostEqual(result["value"], 5.)
        self.assertAlmostEqual(result["coverage"], 4 / 6)
        np.testing.assert_array_equal(values, before)
        unavailable = regional_summary({"metric": np.full((2, 2), np.nan)}, [0])["metric"]
        self.assertTrue(np.isnan(unavailable["value"]))
        self.assertEqual(unavailable["coverage"], 0.)
        aggregated = aggregate_regional_takes([{"metric": {"value": v, "coverage": c}} for v, c in [(0, .2), (1, .3), (9, .7)]])
        self.assertEqual(aggregated["metric"], {"value": 1., "coverage": .3})
        with self.assertRaises(ValueError):
            regional_summary({"metric": values}, [0, 0])

    def test_pair_within_visit_before_participant_median(self):
        contrasts = []
        for visit, a, b in [(1, 0., 0.), (2, 10., 9.), (3, 11., 100.)]:
            left = dict.fromkeys(("reduced_movement", "counter_direction", "excess_movement"), a)
            right = dict.fromkeys(left, b)
            contrasts.append((visit, side_contrasts(left, right)))
        summary = summarize_contrasts(list(reversed(contrasts)))
        self.assertEqual(summary["participant_median"]["reduced_movement"], 0.)
        self.assertEqual(summary["earliest_visit"]["reduced_movement"], 0.)
        self.assertEqual(contrasts[2][1]["excess_movement"], 89.)
        missing = summarize_contrasts([(1, {"metric": np.nan}), (2, {"metric": 2.})], metrics=("metric",))
        self.assertTrue(np.isnan(missing["earliest_visit"]["metric"]))
        self.assertEqual(missing["participant_median"]["metric"], 2.)

    def test_fixed_numerical_ties_and_missing_eligibility_are_distinct(self):
        tie=NUMERICAL_TIE_TOLERANCE_MM
        above=np.nextafter(tie,np.inf); below=np.nextafter(tie,0.)
        values=np.array([[tie,tie,tie],[above,above,above],[below,above,above],
                         [-above,np.nan,above],[np.nan,np.nan,np.nan],[0.,0.,0.]])
        result = pattern_frequencies(values)
        self.assertEqual(tie,1e-6)
        self.assertEqual(result["reduced_movement"]["n_eligible"],5)
        self.assertEqual(result["reduced_movement"]["n_supporting"],1)
        self.assertEqual(result["counter_direction"]["n_eligible"],4)
        self.assertEqual(result["counter_direction"]["n_supporting"],2)
        self.assertEqual(result["excess_movement"]["n_eligible"],5)
        self.assertEqual(result["excess_movement"]["n_supporting"],3)
        self.assertEqual(result["all_metrics"]["n_eligible"],4)
        self.assertEqual(result["all_metrics"]["n_supporting"], 1)
        self.assertEqual(result["all_metrics"]["fraction"],1/4)
        np.testing.assert_allclose(result["all_metrics"]["wilson_interval"],wilson_interval(1,4))
        unavailable=pattern_frequencies(np.full((1,3),np.nan))["all_metrics"]
        self.assertEqual(unavailable["n_eligible"],0)
        self.assertEqual(unavailable["n_supporting"],0)
        self.assertTrue(np.isnan(unavailable["fraction"]))
        self.assertTrue(np.isnan(unavailable["wilson_interval"]).all())
        low, high = wilson_interval(0, 1)
        self.assertAlmostEqual(low, 0., places=12)
        self.assertGreater(high, .5)
        self.assertTrue(np.isnan(wilson_interval(0, 0)).all())


if __name__ == "__main__":
    unittest.main()
