from aps_cp_sat.preprocess.grade_catalog import MergedGradeRuleCatalog
from aps_cp_sat.preprocess.order_preparation import prepare_orders

__all__ = ["prepare_orders_for_model", "MergedGradeRuleCatalog", "prepare_orders"]


def __getattr__(name: str):
    if name == "prepare_orders_for_model":
        from aps_cp_sat.preprocess.pipeline import prepare_orders_for_model

        return prepare_orders_for_model
    raise AttributeError(name)
