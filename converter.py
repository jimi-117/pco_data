import kagglehub
import pymongo
from pymongo import MongoClient
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import shutil
import yaml

class LVIStoYOLOConverter:
    def __init__(self, lvis_path, output_path):
        self.lvis_path = Path(lvis_path)
        self.output_path = Path(output_path)
        self.images_path = self.lvis_path / 'images'
        self.annotations_path = self.lvis_path / 'annotations'
        
        # Create directories for YOLO format
        self.yolo_images = self.output_path / 'images'
        self.yolo_labels = self.output_path / 'labels'
        self.yolo_images.mkdir(parents=True, exist_ok=True)
        self.yolo_labels.mkdir(parents=True, exist_ok=True)

    def load_annotations(self):
        """Load LVIS annotations"""
        with open(self.annotations_path / 'annotations.json', 'r') as f:
            return json.load(f)

    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert from LVIS format [x, y, width, height]
        to YOLO format [x_center, y_center, width, height] (normalized)
        """
        x, y, w, h = bbox
        return [
            (x + w/2) / img_width,  # x_center
            (y + h/2) / img_height, # y_center
            w / img_width,          # width
            h / img_height          # height
        ]

    def convert_dataset(self):
        """Convert dataset to YOLO format"""
        annotations = self.load_annotations()
        
        # Create mapping of category ID to label
        categories = {cat['id']: cat['name'] for cat in annotations['categories']}
        
        # Create mapping of image ID to size
        image_info = {img['id']: (img['width'], img['height'], img['file_name']) 
                     for img in annotations['images']}

        # Process annotations
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            img_width, img_height, img_name = image_info[img_id]
            
            # Convert bounding box
            yolo_bbox = self.convert_bbox_to_yolo(
                ann['bbox'], img_width, img_height
            )
            
            # Create YOLO format label
            label_line = f"0 {' '.join([str(x) for x in yolo_bbox])}\n"
            
            # Save image and label
            src_img_path = self.images_path / img_name
            dst_img_path = self.yolo_images / img_name
            label_path = self.yolo_labels / (Path(img_name).stem + '.txt')
            
            # Copy image
            if src_img_path.exists():
                shutil.copy(str(src_img_path), str(dst_img_path))
                
                # Save label
                with open(label_path, 'a') as f:
                    f.write(label_line)

    def create_yaml(self):
        """Create YOLO configuration file"""
        yaml_content = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'fruit_vegetable'  # Single class
            }
        }
        
        with open(self.output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f)

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """Split dataset"""
        all_images = list(self.yolo_images.glob('*.jpg'))
        np.random.shuffle(all_images)
        
        n = len(all_images)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        
        # Create directories for splits
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.yolo_images / split).mkdir(exist_ok=True)
            (self.yolo_labels / split).mkdir(exist_ok=True)

        # Move files
        for i, img_path in enumerate(all_images):
            if i < train_n:
                dst_dir = 'train'
            elif i < train_n + val_n:
                dst_dir = 'val'
            else:
                dst_dir = 'test'
                
            # Move image
            shutil.move(
                str(img_path),
                str(self.yolo_images / dst_dir / img_path.name)
            )
            
            # Move label
            label_path = self.yolo_labels / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.move(
                    str(label_path),
                    str(self.yolo_labels / dst_dir / label_path.name)
                )



class DataPipeline:
    def __init__(self):
        # MongoDB settings
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['food_detection_db']
        self.collection = self.db['annotations']
        
        # SQL database settings
        self.sql_engine = create_engine('postgresql://user:password@localhost:5432/food_db')
        
        # Logging settings
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def download_from_kaggle(self):
        """
        1. API Source: Download dataset using Kaggle API
        """
        try:
            self.logger.info("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download(
                "henningheyen/lvis-fruits-and-vegetables-dataset"
            )
            self.logger.info(f"Dataset downloaded to: {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {str(e)}")
            raise


    def process_annotations(self, path):
        """
        2. File System Source: Process downloaded file
        """
        try:
            annotations_path = Path(path) / 'annotations' / 'annotations.json'
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to process annotations: {str(e)}")
            raise


    def store_in_mongodb(self, data):
        """
        3. Big Data Source: Store data in MongoDB
        """
        try:
            # Add metadata
            data['metadata'] = {
                'import_date': datetime.now(),
                'source': 'kaggle-lvis-fruits-vegetables',
                'version': '1.0'
            }
            
            # Store in MongoDB
            self.collection.insert_one(data)
            self.logger.info("Data stored in MongoDB successfully")
            
            # Create indexes
            self.collection.create_index([('images.file_name', 1)])
            self.collection.create_index([('annotations.image_id', 1)])
            
        except Exception as e:
            self.logger.error(f"Failed to store in MongoDB: {str(e)}")
            raise


    def create_sql_tables(self, data):
        """
        4. SQL Source: Create metadata tables in PostgreSQL
        """
        try:
            # Convert image metadata to DataFrame
            images_df = pd.DataFrame(data['images'])
            categories_df = pd.DataFrame(data['categories'])
            
            # Save as tables in SQL
            images_df.to_sql('images', self.sql_engine, if_exists='replace', index=False)
            categories_df.to_sql('categories', self.sql_engine, if_exists='replace', index=False)
            
            self.logger.info("SQL tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create SQL tables: {str(e)}")
            raise


    def create_yolo_dataset(self, data, path):
        """
        5. File System Output: Create YOLO format dataset
        """
        try:
            converter = LVIStoYOLOConverter(path, 'yolo_dataset')
            converter.convert_dataset()
            converter.split_dataset()
            converter.create_yaml()
            
            self.logger.info("YOLO dataset created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create YOLO dataset: {str(e)}")
            raise


    def run_pipeline(self):
        """
        Execute pipeline
        """
        try:
            # 1. Kaggle API
            path = self.download_from_kaggle()
            
            # 2. Annotation data
            data = self.process_annotations(path)
            
            # 3. MongoDB integration
            self.store_in_mongodb(data)
            
            # 4. Metadata integration
            self.create_sql_tables(data)
            
            # 5. Creation of YOLO dataset
            self.create_yolo_dataset(data, path)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.mongo_client.close()