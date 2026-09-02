import unittest
import tempfile
import os
from pathlib import Path

from src.ingestion.dataset_loader import PCBDefectDatasetLoader, DIE_DEFECT_CLASSES

class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create mock PCB dataset hierarchy using portable format
        for cls in DIE_DEFECT_CLASSES:
            cls_dir = self.root / "PCB_DATASET" / "images" / cls.capitalize()
            cls_dir.mkdir(parents=True, exist_ok=True)

            ann_dir = self.root / "PCB_DATASET" / "Annotations" / cls.capitalize()
            ann_dir.mkdir(parents=True, exist_ok=True)

            for i in range(12):
                img_path = cls_dir / f"{cls}_{i:02d}.png"
                # Write minimal valid dummy binary
                with open(img_path, "wb") as f:
                    f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

                xml_path = ann_dir / f"{cls}_{i:02d}.xml"
                xml_content = f"""<annotation>
                    <folder>{cls.capitalize()}</folder>
                    <filename>{cls}_{i:02d}.png</filename>
                    <size><width>300</width><height>300</height><depth>3</depth></size>
                    <object>
                        <name>{cls}</name>
                        <bndbox>
                            <xmin>50</xmin>
                            <ymin>50</ymin>
                            <xmax>150</xmax>
                            <ymax>150</ymax>
                        </bndbox>
                    </object>
                </annotation>"""
                with open(xml_path, "w") as f:
                    f.write(xml_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_discovery_and_k_shot_split(self):
        loader = PCBDefectDatasetLoader(data_dir=str(self.root / "PCB_DATASET"), k_shot=5, val_ratio=0.2)
        train_samples, val_samples, test_samples = loader.load_datasets()
        self.assertEqual(len(train_samples), 30)
        self.assertGreater(len(val_samples), 0)
        self.assertGreater(len(test_samples), 0)
