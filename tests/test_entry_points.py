import ast
from pathlib import Path

ENTRY_POINTS = (
    "train_ssl",
    "extract_embeddings",
    "train_mil",
    "linear_probe",
    "build_camelyon16_metadata",
    "make_tcga_nsclc_split",
    "build_tcga_nsclc_metadata",
)


def test_entry_points_have_main_guard() -> None:
    for name in ENTRY_POINTS:
        tree = ast.parse((Path("src") / f"{name}.py").read_text())
        guards = [node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)]
        assert guards, f"src/{name}.py lacks an `if __name__ == '__main__'` guard"
