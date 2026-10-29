from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

D = Decimal
JUDGES = ["GPT-4o", "Claude-3.5-Sonnet", "Llama-3-70B"]
CONFIGURATIONS = ["Standard Prompt", "Clinical-HELM Prompt", "RCSA"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot export an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: str(value) if isinstance(value, D) else value for key, value in row.items()}
            )


def preference_bounds(
    mean_difference: Decimal,
    width: Decimal = D("5"),
    epsilon: Decimal = D("0.01"),
    margin: Decimal = D("0"),
) -> tuple[Decimal, Decimal]:
    values = [mean_difference, width, epsilon, margin]
    if not all(value.is_finite() for value in values):
        raise ValueError("All bound inputs must be finite")
    if width <= 0 or epsilon < 0 or not D(0) <= margin < width:
        raise ValueError("Require width > 0, epsilon >= 0 and 0 <= margin < width")
    if abs(mean_difference) > width + epsilon:
        raise ValueError("Mean difference is incompatible with the proposed support")
    lo_mean = max(-width, mean_difference - epsilon)
    hi_mean = min(width, mean_difference + epsilon)
    lower = max(D(0), (lo_mean - margin) / (width - margin))
    upper = min(D(1), (width + hi_mean) / (width + margin))
    assert D(0) <= lower <= upper <= D(1)
    return lower, upper


def software_checks() -> dict[str, Any]:
    point_checks = 0
    for i in range(11):
        for j in range(11):
            difference = D(i - j) / 2
            for margin in [D(0), D("0.5"), D(1), D(2), D("4.5")]:
                lower, upper = preference_bounds(difference, epsilon=D(0), margin=margin)
                event = D(int(difference > margin))
                assert lower <= event <= upper
                point_checks += 1
    generator = random.Random(20261029)
    fixture_checks = 0
    for _ in range(200):
        left = [D(generator.randrange(501)) / 100 for _ in range(40)]
        right = [D(generator.randrange(501)) / 100 for _ in range(40)]
        printed_difference = (sum(left) / 40).quantize(D("0.01"), rounding=ROUND_HALF_UP) - (
            sum(right) / 40
        ).quantize(D("0.01"), rounding=ROUND_HALF_UP)
        for margin in [D(0), D(1), D(2), D(3)]:
            lower, upper = preference_bounds(printed_difference, margin=margin)
            observed_in_fixture = sum(a - b > margin for a, b in zip(left, right)) / D(40)
            assert lower <= observed_in_fixture <= upper
            fixture_checks += 1
    for width, epsilon, margin in [
        (D(0), D(0), D(0)),
        (D(5), D("-0.01"), D(0)),
        (D(5), D(0), D(5)),
    ]:
        try:
            preference_bounds(D(0), width, epsilon, margin)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid inputs were accepted")
    return {
        "pointwise_grid_checks": point_checks,
        "rounded_distribution_fixture_checks": fixture_checks,
        "invalid_parameter_checks": 3,
        "fixture_seed": 20261029,
        "fixtures_are_study_data": False,
    }


def analyse(source: Path, output: Path) -> dict[str, Any]:
    scores = read_rows(source / "judge_scores.csv")
    agreement = read_rows(source / "agreement.csv")
    assert len(scores) == 9 and len(agreement) == 3
    indexed = {(row["judge_model"], row["prompt_configuration"]): row for row in scores}
    assert set(indexed) == {(judge, config) for judge in JUDGES for config in CONFIGURATIONS}
    full_scores, changes, decomposition, combined_changes = [], [], [], []
    bounds, margins, eps_samples, eps_curve, support_samples, support_curve = [], [], [], [], [], []
    thresholds, normalizations, omissions, agreements = [], [], [], []
    for judge in JUDGES:
        for config in CONFIGURATIONS:
            row = indexed[judge, config]
            refusal = D(row["safe_refusal_score"])
            hallucinated = D(row["hallucinated_response_score"])
            delta = hallucinated - refusal
            assert D(row["delta_hallucinated_minus_refusal"]) == delta
            assert all(D(0) <= score <= D(5) for score in [refusal, hallucinated])
            mu = -delta
            full_scores.append(
                {
                    "judge": judge,
                    "configuration": config,
                    "refusal_mean": refusal,
                    "hallucinated_mean": hallucinated,
                    "hallucinated_minus_refusal": delta,
                    "contrast_rounding_lower": delta - D("0.01"),
                    "contrast_rounding_upper": delta + D("0.01"),
                    "rounding_interval_excludes_zero": abs(delta) > D("0.01"),
                }
            )
            for margin in [D(0), D(1), D(2)]:
                lower, upper = preference_bounds(mu, margin=margin)
                bounds.append(
                    {
                        "judge": judge,
                        "configuration": config,
                        "margin": margin,
                        "mean_refusal_minus_hallucinated": mu,
                        "support_width": D(5),
                        "difference_rounding_tolerance": D("0.01"),
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
            for index in range(99):
                margin = D(index) / 20
                lower, upper = preference_bounds(mu, margin=margin)
                margins.append(
                    {
                        "judge": judge,
                        "configuration": config,
                        "margin": margin,
                        "support_width": D(5),
                        "difference_rounding_tolerance": D("0.01"),
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
            for scale, factor in [("0 to 1", D("0.2")), ("0 to 5", D(1)), ("0 to 100", D(20))]:
                normalizations.append(
                    {
                        "judge": judge,
                        "configuration": config,
                        "display_scale": scale,
                        "scale_factor": factor,
                        "refusal_mean_rescaled": refusal * factor,
                        "hallucinated_mean_rescaled": hallucinated * factor,
                        "gap_rescaled": delta * factor,
                    }
                )
            if config == "RCSA":
                thresholds.append(
                    {
                        "judge": judge,
                        "mean_refusal_minus_hallucinated": mu,
                        "support_width": D(5),
                        "difference_rounding_tolerance": D("0.01"),
                        "majority_margin_limit": 2 * (mu - D("0.01")) - 5,
                        "support_width_limit_at_margin_2": 2 * (mu - D("0.01")) - 2,
                        "threshold_relation": "Strictly less than the limit",
                    }
                )
                for epsilon in [D(0), D("0.01"), D("0.02"), D("0.05")]:
                    for margin in [D(0), D(1), D(2)]:
                        lower, upper = preference_bounds(mu, epsilon=epsilon, margin=margin)
                        eps_samples.append(
                            {
                                "judge": judge,
                                "difference_rounding_tolerance": epsilon,
                                "margin": margin,
                                "support_width": D(5),
                                "lower_bound": lower,
                                "upper_bound": upper,
                            }
                        )
                for index in range(51):
                    epsilon = D(index) / 1000
                    lower, upper = preference_bounds(mu, epsilon=epsilon, margin=D(2))
                    eps_curve.append(
                        {
                            "judge": judge,
                            "difference_rounding_tolerance": epsilon,
                            "margin": D(2),
                            "support_width": D(5),
                            "lower_bound": lower,
                            "upper_bound": upper,
                        }
                    )
                for width in [D(5), D("5.25"), D("5.5"), D(6), D("7.5"), D(10)]:
                    for margin in [D(0), D(1), D(2)]:
                        lower, upper = preference_bounds(mu, width=width, margin=margin)
                        support_samples.append(
                            {
                                "judge": judge,
                                "support_width": width,
                                "margin": margin,
                                "difference_rounding_tolerance": D("0.01"),
                                "lower_bound": lower,
                                "upper_bound": upper,
                            }
                        )
                for index in range(101):
                    width = 5 + D(index) / 20
                    lower, upper = preference_bounds(mu, width=width, margin=D(2))
                    support_curve.append(
                        {
                            "judge": judge,
                            "support_width": width,
                            "margin": D(2),
                            "difference_rounding_tolerance": D("0.01"),
                            "lower_bound": lower,
                            "upper_bound": upper,
                        }
                    )
        standard = indexed[judge, "Standard Prompt"]
        clinical = indexed[judge, "Clinical-HELM Prompt"]
        rcsa = indexed[judge, "RCSA"]
        standard_gap = D(standard["delta_hallucinated_minus_refusal"])
        clinical_gap = D(clinical["delta_hallucinated_minus_refusal"])
        rcsa_gap = D(rcsa["delta_hallucinated_minus_refusal"])
        narrowing = standard_gap - clinical_gap
        change = {
            "judge": judge,
            "standard_gap": standard_gap,
            "clinical_gap": clinical_gap,
            "gap_reduction": narrowing,
            "relative_gap_reduction": narrowing / standard_gap,
            "gap_reduction_rounding_lower": narrowing - D("0.02"),
            "gap_reduction_rounding_upper": narrowing + D("0.02"),
        }
        changes.append(change)
        refusal_increase = D(rcsa["safe_refusal_score"]) - D(standard["safe_refusal_score"])
        hallucinated_drop = D(standard["hallucinated_response_score"]) - D(
            rcsa["hallucinated_response_score"]
        )
        total = standard_gap - rcsa_gap
        assert refusal_increase + hallucinated_drop == total
        component = {
            "judge": judge,
            "refusal_increase": refusal_increase,
            "hallucinated_decrease": hallucinated_drop,
            "total_contrast_shift": total,
            "refusal_component_fraction": refusal_increase / total,
            "hallucinated_component_fraction": hallucinated_drop / total,
            "total_shift_rounding_lower": total - D("0.02"),
            "total_shift_rounding_upper": total + D("0.02"),
        }
        decomposition.append(component)
        combined_changes.append(change | component | thresholds[-1])
    for row in agreement:
        agreements.append(
            {
                "clinical_domain": row["clinical_domain"],
                "triplets_n": int(row["cases_n"]),
                "kappa_without_rcsa": D(row["kappa_without_rcsa"]),
                "kappa_with_rcsa": D(row["kappa_with_rcsa"]),
                "kappa_change": D(row["kappa_with_rcsa"]) - D(row["kappa_without_rcsa"]),
                "reported_p_value": D(row["p_value"]),
                "p_value_status": "Retained source value; test provenance unavailable",
            }
        )
    assert sum(row["triplets_n"] for row in agreements) == 1200
    for omitted in ["None"] + JUDGES:
        for config in CONFIGURATIONS:
            values = [
                D(indexed[judge, config]["delta_hallucinated_minus_refusal"])
                for judge in JUDGES
                if judge != omitted
            ]
            omissions.append(
                {
                    "omitted_judge": omitted,
                    "configuration": config,
                    "judges_retained": len(values),
                    "unweighted_judge_mean_gap": sum(values) / len(values),
                    "minimum_judge_gap": min(values),
                    "maximum_judge_gap": max(values),
                    "all_gaps_have_same_sign": all(value > 0 for value in values)
                    or all(value < 0 for value in values),
                }
            )
    result_tables = {
        "scores_full.csv": full_scores,
        "prompt_changes.csv": changes,
        "contrast_components.csv": decomposition,
        "score_changes.csv": combined_changes,
        "agreement_changes.csv": agreements,
        "bounds_27.csv": bounds,
        "margin_curve.csv": margins,
        "rounding_sensitivity.csv": eps_samples,
        "rounding_curve.csv": eps_curve,
        "support_sensitivity.csv": support_samples,
        "support_curve.csv": support_curve,
        "majority_thresholds.csv": thresholds,
        "score_normalization.csv": normalizations,
        "judge_omission.csv": omissions,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in result_tables.items():
        write_rows(output / name, rows)
    tests = software_checks()
    assert len(bounds) == 27 and len(margins) == 891
    assert [row["majority_margin_limit"] for row in thresholds] == [D("2.44"), D("2.68"), D("2.14")]
    for row in full_scores:
        assert row["rounding_interval_excludes_zero"]
    report = {
        "inputs": {
            name: {
                "rows": len(read_rows(source / name)),
                "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
            }
            for name in ["judge_scores.csv", "agreement.csv"]
        },
        "output_row_counts": {name: len(rows) for name, rows in result_tables.items()},
        "arithmetic_decompositions_verified": 3,
        "source_values_preserved": True,
        "software_checks": tests,
        "new_model_or_clinician_observations": 0,
        "analytical_bounds_are_sampling_intervals": False,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce aggregate clinical-judge calculations and conditional bounds"
    )
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    arguments = parser.parse_args()
    report = analyse(arguments.data.resolve(), arguments.output.resolve())
    print(
        json.dumps(
            {
                "inputs": [9, 3],
                "result_tables": len(report["output_row_counts"]),
                "software_checks": report["software_checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
