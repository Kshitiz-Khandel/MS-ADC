import unittest
import tempfile
import os
from pathlib import Path
from PIL import Image
from src.ingestion.dataset_loader import PCBDefectDatasetLoader, DIE_DEFECT_CLASSES
from src.ingestion.augmentor import MetrologyAugmentor


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create mock PCB dataset hierarchy
        for cls in DIE_DEFECT_CLASSES:
            cls_dir = self.root / "PCB_DATASET" / "images" / cls.capitalize()
            cls_dir.mkdir(parents=True, exist_ok=True)

            ann_dir = self.root / "PCB_DATASET" / "Annotations" / cls.capitalize()
            ann_dir.mkdir(parents=True, exist_ok=True)

            for i in range(12):
                img_path = cls_dir / f"{cls}_{i:02d}.jpg"
                img = Image.new("RGB", (300, 300), color=(i * 10, i * 20, i * 15))
                img.save(img_path)

                # Write mock Pascal VOC XML
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

    def test_discover_images(self):
        loader = PCBDefectDatasetLoader(data_dir=self.root)
        discovered = loader.discover_image_files()

        for cls in DIE_DEFECT_CLASSES:
            self.assertIn(cls, discovered)
            self.assertEqual(len(discovered[cls]), 12)

    def test_load_and_preprocess_image(self):
        loader = PCBDefectDatasetLoader(data_dir=self.root, crop_padding=10)
        discovered = loader.discover_image_files()

        img_path = discovered["short"][0]
        patch = loader.load_and_preprocess_image(img_path)
        self.assertIsInstance(patch, Image.Image)
        self.assertEqual(patch.size, (224, 224))

    def test_stratified_split(self):
        loader = PCBDefectDatasetLoader(data_dir=self.root)
        train_s, val_s, test_s = loader.get_stratified_split(k_shot_train=6, val_ratio=0.5, seed=42)

        for cls in DIE_DEFECT_CLASSES:
            self.assertEqual(len(train_s[cls]), 6)
            self.assertEqual(len(val_s[cls]), 3)
            self.assertEqual(len(test_s[cls]), 3)

    def test_augmentor_pil(self):
        aug = MetrologyAugmentor(target_size=224)
        sample = Image.new("RGB", (224, 224), color=(128, 64, 32))
        res = aug.augment_pil_image(sample)
        self.assertIsInstance(res, Image.Image)
        self.assertEqual(res.size, (224, 224))


if __name__ == "__main__":
    unittest.main()
