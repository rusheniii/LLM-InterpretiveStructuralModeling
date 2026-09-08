import pickle
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import networkx as nx

import llmism.calculate_shd as shd


def _factor(name: str):
    return SimpleNamespace(name=name)


class TestCalculateShd(TestCase):
    def test_build_ssim_digraph_adds_edges_by_relationship_symbol(self) -> None:
        edges = [(_factor("A"), _factor("B"), "V"), (_factor("B"), _factor("C"), "A"), (_factor("A"), _factor("C"), "X")]
        result = shd.build_ssim_digraph({"A", "B", "C"}, edges)
        self.assertEqual(set(result.edges()), {("A", "B"), ("C", "B"), ("A", "C"), ("C", "A")})

    def test_build_transitive_closure_adds_reachable_and_self_edges(self) -> None:
        result = shd.build_transitive_closure(nx.DiGraph([("A", "B"), ("B", "C")]))
        self.assertEqual(set(result.edges()), {("A", "A"), ("A", "B"), ("A", "C"), ("B", "B"), ("B", "C"), ("C", "C")})

    def test_nx_to_np_array_sorts_nodes_when_not_provided(self) -> None:
        graph = nx.DiGraph([("B", "B"), ("A", "A"), ("A", "B")])
        self.assertEqual(shd.nx_to_np_array(graph).tolist(), [[1, 1], [0, 1]])

    def test_nx_to_np_array_uses_provided_nodelist_order(self) -> None:
        graph = nx.DiGraph([("B", "B"), ("A", "A"), ("A", "B")])
        self.assertEqual(shd.nx_to_np_array(graph, nodelist=["B", "A"]).tolist(), [[1, 0], [1, 1]])

    def test_validate_graphs_rejects_mismatched_graph_types(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "type of graphs"):
            shd.validate_graphs(nx.Graph(), nx.DiGraph())

    def test_validate_graphs_rejects_mismatched_nodes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "nodes of the graphs"):
            shd.validate_graphs(nx.DiGraph([("A", "A")]), nx.DiGraph([("B", "B")]))

    def test_load_true_graph_reads_csvs_and_builds_transitive_closure(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            barriers = base / "barriers.csv"
            reachability = base / "final_reachability_matrix.csv"
            barriers.write_text("Code,Barrier\n1,A\n2,B\n3,C\n", encoding="utf-8")
            reachability.write_text("Code,1,2,3\n1,0,1,0\n2,0,0,1*\n3,0,0,0\n", encoding="utf-8")
            result = shd.load_true_graph(barriers, reachability)
        self.assertEqual(set(result.edges()), {("A", "A"), ("A", "B"), ("A", "C"), ("B", "B"), ("B", "C"), ("C", "C")})

    def test_load_true_graph_skips_unknown_codes_columns_and_diagonal(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            barriers = base / "barriers.csv"
            reachability = base / "final_reachability_matrix.csv"
            barriers.write_text("Code,Barrier\nB1,A\nB2,B\n", encoding="utf-8")
            reachability.write_text(
                "Code,B1,B2,B9\nB1,1,1,1\nB2,0,1,1\nB9,1,1,1\n",
                encoding="utf-8",
            )
            result = shd.load_true_graph(barriers, reachability)
        self.assertEqual(set(result.edges()), {("A", "A"), ("A", "B"), ("B", "B")})

    def test_calculate_metric_returns_hamming_loss(self) -> None:
        true_graph = nx.DiGraph([("A", "A"), ("A", "B"), ("B", "B")])
        predicted_graph = nx.DiGraph([("A", "B")])
        predicted_graph.add_nodes_from(["A", "B"])
        self.assertEqual(shd.calculate_metric(predicted_graph, true_graph, "shd"), 0.0)

    def test_calculate_metric_returns_f1_score(self) -> None:
        graph = nx.DiGraph([("A", "A"), ("A", "B"), ("B", "B")])
        self.assertEqual(shd.calculate_metric(graph, graph, "f1"), 1.0)

    def test_calculate_metric_returns_precision_score(self) -> None:
        graph = nx.DiGraph([("A", "A"), ("A", "B"), ("B", "B")])
        self.assertEqual(shd.calculate_metric(graph, graph, "tp_nnz"), 1.0)

    def test_load_ssim_edges_returns_none_for_missing_file(self) -> None:
        self.assertIsNone(shd.load_ssim_edges("missing-file.bin"))

    def test_load_ssim_edges_reads_pickle_file(self) -> None:
        expected = [("edge",)]
        with TemporaryDirectory() as directory:
            graph_file = Path(directory) / "graph.adjlist"
            with graph_file.open("wb") as file:
                pickle.dump(expected, file)
            result = shd.load_ssim_edges(graph_file)
        self.assertEqual(result, expected)

    def test_load_ssim_edges_reraises_unpickling_errors(self) -> None:
        with TemporaryDirectory() as directory:
            graph_file = Path(directory) / "graph.adjlist"
            graph_file.write_bytes(b"not-a-pickle")
            with patch("builtins.print") as print_mock:
                with self.assertRaises(pickle.UnpicklingError):
                    shd.load_ssim_edges(graph_file)
        self.assertEqual(print_mock.call_count, 1)

    def test_main_calculates_and_prints_all_metrics(self) -> None:
        graph = nx.DiGraph([("A", "A")])
        with (
            patch.object(shd, "load_true_graph", return_value=graph) as load_true,
            patch.object(shd, "load_ssim_edges", return_value=graph) as load_ssim,
            patch.object(shd, "calculate_metric", side_effect=[0.0, 1.0, 1.0]) as calculate,
            patch("builtins.print") as print_mock,
        ):
            shd.main("project")
        load_true.assert_called_once_with(Path("project") / "barriers.csv", Path("project") / "final_reachability_matrix.csv")
        load_ssim.assert_called_once_with(Path("project") / "graph.adjlist")
        self.assertEqual(calculate.call_count, 3)
        self.assertEqual(print_mock.call_count, 4)

    def test_cli_runs_main_with_parsed_base_path(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "barriers.csv").write_text("Code,Barrier\n1,A\n2,B\n", encoding="utf-8")
            (base / "final_reachability_matrix.csv").write_text("Code,1,2\n1,0,1\n2,0,0\n", encoding="utf-8")
            graph = nx.DiGraph([("A", "B")])
            graph.add_nodes_from(["A", "B"])
            with (base / "graph.adjlist").open("wb") as file:
                pickle.dump(graph, file)
            with (
                patch("sys.argv", ["calculate_shd.py", str(base)]),
                patch("builtins.print") as print_mock,
            ):
                runpy.run_module("llmism.calculate_shd", run_name="__main__")
        self.assertEqual(print_mock.call_count, 4)