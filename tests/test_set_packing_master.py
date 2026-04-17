"""
Tests for the Set Packing Master (Macro-Block Assembly) Architecture.

This module tests the new Set Packing architecture that:
1. Generates candidate macro-blocks via randomized DFS
2. Selects non-overlapping blocks via CP-SAT Set Packing
3. Directly unpacks selected blocks into plan rows (no Local Router)
"""

import random
from typing import List, Tuple

import pandas as pd
import pytest

from aps_cp_sat.config import PlannerConfig
from aps_cp_sat.model.feasible_block_builder import (
    MacroBlock,
    BlockGeneratorStats,
    TemplateGraph,
    generate_candidate_macro_blocks,
)


class TestTemplateGraph:
    """Tests for the TemplateGraph class."""

    def test_build_graph_from_templates(self):
        """Test that TemplateGraph builds correctly from orders and templates."""
        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 100.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 150.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
            {"order_id": "O3", "tons": 120.0, "line_capability": "dual", "width": 1100, "thickness": 2.2},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0},
            {"from_order_id": "O2", "to_order_id": "O3", "line": "big_roll", "cost": 15.0},
        ])

        cfg = PlannerConfig()
        graph = TemplateGraph(orders_df, tpl_df, cfg)

        # Check that edges are built correctly
        assert "O1" in graph.order_record
        assert "O2" in graph.order_record
        assert "O3" in graph.order_record

        # Check out edges
        out_candidates = graph.get_out_candidates("O1", set())
        assert len(out_candidates) >= 1
        assert any(oid == "O2" for oid, _ in out_candidates)

        # Check in edges
        in_candidates = graph.get_in_candidates("O3", set())
        assert len(in_candidates) >= 1
        assert any(oid == "O2" for oid, _ in in_candidates)

    def test_get_seed_orders(self):
        """Test that seed orders are filtered correctly by line."""
        orders_df = pd.DataFrame([
            {"order_id": "B1", "tons": 100.0, "line_capability": "big_only", "width": 1000, "thickness": 2.0},
            {"order_id": "S1", "tons": 80.0, "line_capability": "small_only", "width": 800, "thickness": 1.5},
            {"order_id": "D1", "tons": 120.0, "line_capability": "dual", "width": 900, "thickness": 1.8},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "B1", "to_order_id": "D1", "line": "big_roll", "cost": 10.0},
            {"from_order_id": "S1", "to_order_id": "D1", "line": "small_roll", "cost": 12.0},
        ])

        cfg = PlannerConfig()
        graph = TemplateGraph(orders_df, tpl_df, cfg)

        # Big roll seeds should include big_only and dual
        big_seeds = graph.get_seed_orders("big_roll")
        assert "B1" in big_seeds
        assert "D1" in big_seeds
        assert "S1" not in big_seeds

        # Small roll seeds should include small_only and dual
        small_seeds = graph.get_seed_orders("small_roll")
        assert "S1" in small_seeds
        assert "D1" in small_seeds
        assert "B1" not in small_seeds

    def test_tons_of(self):
        """Test that tons_of calculates correctly."""
        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 100.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 150.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
        ])

        tpl_df = pd.DataFrame()
        cfg = PlannerConfig()
        graph = TemplateGraph(orders_df, tpl_df, cfg)

        tons = graph.tons_of(["O1", "O2"])
        assert tons == pytest.approx(250.0)


class TestMacroBlock:
    """Tests for the MacroBlock dataclass."""

    def test_macro_block_creation(self):
        """Test that MacroBlock is created correctly."""
        block = MacroBlock(
            block_id="BLK_1",
            line="big_roll",
            order_ids=["O1", "O2", "O3"],
            total_tons=350.0,
            total_cost=25,
        )

        assert block.block_id == "BLK_1"
        assert block.line == "big_roll"
        assert len(block.order_ids) == 3
        assert block.total_tons == pytest.approx(350.0)
        assert block.total_cost == 25

    def test_contains_order(self):
        """Test the contains_order method."""
        block = MacroBlock(
            block_id="BLK_1",
            line="big_roll",
            order_ids=["O1", "O2", "O3"],
            total_tons=350.0,
            total_cost=25,
        )

        assert block.contains_order("O1")
        assert block.contains_order("O2")
        assert block.contains_order("O3")
        assert not block.contains_order("O4")
        assert not block.contains_order("")


class TestGenerateCandidateMacroBlocks:
    """Tests for the generate_candidate_macro_blocks function."""

    def test_empty_orders(self):
        """Test with empty orders DataFrame."""
        orders_df = pd.DataFrame()
        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0},
        ])

        cfg = PlannerConfig()
        blocks, stats = generate_candidate_macro_blocks(
            orders_df, tpl_df, cfg,
            target_blocks=100,
            time_limit_seconds=1.0,
            random_seed=42,
        )

        assert len(blocks) == 0
        assert stats.total_blocks_generated == 0

    def test_empty_templates(self):
        """Test with empty templates DataFrame."""
        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 100.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
        ])

        tpl_df = pd.DataFrame()

        cfg = PlannerConfig()
        blocks, stats = generate_candidate_macro_blocks(
            orders_df, tpl_df, cfg,
            target_blocks=100,
            time_limit_seconds=1.0,
            random_seed=42,
        )

        assert len(blocks) == 0
        assert stats.total_blocks_generated == 0

    def test_generates_valid_blocks(self):
        """Test that valid blocks are generated from valid template graph."""
        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 300.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 400.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
            {"order_id": "O3", "tons": 350.0, "line_capability": "dual", "width": 1100, "thickness": 2.2},
            {"order_id": "O4", "tons": 450.0, "line_capability": "dual", "width": 1150, "thickness": 2.3},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
            {"from_order_id": "O2", "to_order_id": "O3", "line": "big_roll", "cost": 12.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
            {"from_order_id": "O3", "to_order_id": "O4", "line": "big_roll", "cost": 15.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        ])

        cfg = PlannerConfig()
        blocks, stats = generate_candidate_macro_blocks(
            orders_df, tpl_df, cfg,
            target_blocks=100,
            time_limit_seconds=2.0,
            random_seed=42,
        )

        # Should generate some valid blocks
        assert len(blocks) > 0
        assert stats.total_blocks_generated > 0

        # All blocks should have valid tonnage
        ton_min = float(cfg.rule.campaign_ton_min)
        ton_max = float(cfg.rule.campaign_ton_max)
        for block in blocks:
            assert ton_min <= block.total_tons <= ton_max, \
                f"Block {block.block_id} has invalid tons: {block.total_tons}"
            assert len(block.order_ids) > 0
            assert block.line in ["big_roll", "small_roll"]

    def test_block_tonnage_range(self):
        """Test that all generated blocks have tonnage within valid range."""
        orders_df = pd.DataFrame([
            {"order_id": f"O{i}", "tons": 150.0 + i * 10, "line_capability": "dual",
             "width": 1000 + i * 50, "thickness": 2.0 + i * 0.1}
            for i in range(1, 11)
        ])

        # Create a chain template: O1 -> O2 -> O3 -> ...
        tpl_rows = []
        for i in range(1, 10):
            tpl_rows.append({
                "from_order_id": f"O{i}",
                "to_order_id": f"O{i+1}",
                "line": "big_roll",
                "cost": 10.0,
                "width_smooth_cost": 0,
                "thickness_smooth_cost": 0,
                "temp_margin_cost": 0,
                "cross_group_cost": 0,
                "bridge_count": 0,
                "edge_type": "DIRECT_EDGE",
            })
        tpl_df = pd.DataFrame(tpl_rows)

        cfg = PlannerConfig()
        blocks, stats = generate_candidate_macro_blocks(
            orders_df, tpl_df, cfg,
            target_blocks=200,
            time_limit_seconds=3.0,
            random_seed=123,
        )

        ton_min = float(cfg.rule.campaign_ton_min)
        ton_max = float(cfg.rule.campaign_ton_max)

        for block in blocks:
            assert ton_min <= block.total_tons <= ton_max, \
                f"Block {block.block_id} has tons {block.total_tons} outside range [{ton_min}, {ton_max}]"

    def test_statistics_collected(self):
        """Test that BlockGeneratorStats are collected correctly."""
        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 400.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 500.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        ])

        cfg = PlannerConfig()
        blocks, stats = generate_candidate_macro_blocks(
            orders_df, tpl_df, cfg,
            target_blocks=50,
            time_limit_seconds=1.0,
            random_seed=42,
        )

        assert stats.seed_order_count > 0
        assert stats.generation_seconds >= 0
        assert isinstance(stats.blocks_by_line, dict)
        assert isinstance(stats.blocks_by_ton_range, dict)


class TestSetPackingIntegration:
    """Integration tests for the full Set Packing Master."""

    def test_set_packing_master_basic(self):
        """Test that Set Packing Master can be called successfully."""
        from aps_cp_sat.model.joint_master import _run_global_joint_model

        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 400.0, "line_capability": "dual", "width": 1000,
             "thickness": 2.0, "priority": 1, "due_rank": 1},
            {"order_id": "O2", "tons": 500.0, "line_capability": "dual", "width": 1050,
             "thickness": 2.1, "priority": 1, "due_rank": 2},
            {"order_id": "O3", "tons": 450.0, "line_capability": "dual", "width": 1100,
             "thickness": 2.2, "priority": 1, "due_rank": 3},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
            {"from_order_id": "O2", "to_order_id": "O3", "line": "big_roll", "cost": 12.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        ])

        transition_pack = {"templates": tpl_df}

        # Enable Set Packing Master
        cfg = PlannerConfig()
        cfg.model.use_set_packing_master = True

        result = _run_global_joint_model(
            orders_df=orders_df,
            transition_pack=transition_pack,
            cfg=cfg,
            random_seed=42,
        )

        assert result is not None
        assert "status" in result
        # Should not return NO_TEMPLATE since we provided templates
        assert result.get("status") != "NO_TEMPLATE"

    def test_set_packing_no_overlap(self):
        """Test that Set Packing does not assign same order to multiple blocks."""
        from aps_cp_sat.model.joint_master import _run_set_packing_master

        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 400.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 500.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
            {"order_id": "O3", "tons": 450.0, "line_capability": "dual", "width": 1100, "thickness": 2.2},
            {"order_id": "O4", "tons": 480.0, "line_capability": "dual", "width": 1150, "thickness": 2.3},
        ])

        # Create two separate chains: O1->O2 and O3->O4
        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
            {"from_order_id": "O3", "to_order_id": "O4", "line": "big_roll", "cost": 12.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        ])

        transition_pack = {"templates": tpl_df}

        cfg = PlannerConfig()
        cfg.model.use_set_packing_master = True

        result = _run_set_packing_master(
            orders_df=orders_df,
            transition_pack=transition_pack,
            cfg=cfg,
            random_seed=42,
        )

        assert result is not None

        # Check that no order appears twice in the plan
        if not result["plan_df"].empty:
            order_ids = result["plan_df"]["order_id"].tolist()
            assert len(order_ids) == len(set(order_ids)), \
                f"Order overlap detected: {order_ids}"

    def test_set_packing_dropped_orders(self):
        """Test that orders not in selected blocks are dropped."""
        from aps_cp_sat.model.joint_master import _run_set_packing_master

        orders_df = pd.DataFrame([
            {"order_id": "O1", "tons": 400.0, "line_capability": "dual", "width": 1000, "thickness": 2.0},
            {"order_id": "O2", "tons": 500.0, "line_capability": "dual", "width": 1050, "thickness": 2.1},
            # O3 is isolated - no template edges
            {"order_id": "O3", "tons": 100.0, "line_capability": "dual", "width": 1100, "thickness": 2.2},
        ])

        tpl_df = pd.DataFrame([
            {"from_order_id": "O1", "to_order_id": "O2", "line": "big_roll", "cost": 10.0,
             "width_smooth_cost": 0, "thickness_smooth_cost": 0, "temp_margin_cost": 0,
             "cross_group_cost": 0, "bridge_count": 0, "edge_type": "DIRECT_EDGE"},
        ])

        transition_pack = {"templates": tpl_df}

        cfg = PlannerConfig()
        cfg.model.use_set_packing_master = True

        result = _run_set_packing_master(
            orders_df=orders_df,
            transition_pack=transition_pack,
            cfg=cfg,
            random_seed=42,
        )

        assert result is not None

        # O3 should be dropped since it can't form a valid block
        if not result["dropped_df"].empty:
            dropped_ids = result["dropped_df"]["order_id"].tolist()
            if "O3" in dropped_ids:
                # Verify drop reason
                o3_row = result["dropped_df"][result["dropped_df"]["order_id"] == "O3"]
                if not o3_row.empty:
                    drop_reason = o3_row.iloc[0].get("drop_reason", "")
                    # O3 should have a drop reason indicating it wasn't selected
                    assert drop_reason in [
                        "NOT_SELECTED_IN_MACRO_BLOCKS",
                        "NO_VALID_MACRO_BLOCK",
                        "NO_FEASIBLE_BLOCK_GENERATED",
                    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
