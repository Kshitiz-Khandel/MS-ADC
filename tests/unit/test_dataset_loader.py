import unittest
import tempfile
import os
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.ingestion.dataset_loader import PCBDefectDatasetLoader, DIE_DEFECT_CLASSES
from src.ingestion.augmentor import MetrologyAugmentor

@unittest.skipIf(not HAS_PIL, "PIL not installed in test environment")
class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        for cls in DIE_DEFECT_CLASSES:
            cls_dir = self.root / "PCB_DATASET" / "images" / cls.capitalize()
            cls_dir.mkdir(parents=True, exist_ok=True)

            ann_dir = self.root / "PCB_DATASET" / "Annotations" / cls.capitalize()
            ann_dir.mkdir(parents=True, exist_ok=True)

            for i in range(12):
                img_path = cls_dir / f"{cls}_{i:02d}.jpg"
                img = Image.new("RGB", (300, 300), color=(i * 10, i * 20, i * 15))
                img.save(img_path)

                xml_path = ann_dir / f"{cls}_{i:02d}.xml"
                xml_content = f"""<annotation>
                    <folder>{cls.capitalize()}</folder>
                    <filename>{cls}_{i:02d}.jpg</filename>
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
