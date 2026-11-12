import cellxgene_census

from constants import CENSUS_VERSION, DATA_ROOT


DATASETS = {
    "Hao_PBMC": "ed5d841d-6346-47d4-ab2f-7119ad7e3a35",
}


for name, dataset_id in DATASETS.items():
    output_dir = DATA_ROOT / "downstream_datasets" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cellxgene_census.download_source_h5ad(
        dataset_id=dataset_id,
        to_path=str(output_dir / f"{name}.h5ad"),
        census_version=CENSUS_VERSION,
    )