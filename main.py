# Quick example to grab training data automatically
from bing_image_downloader import downloader

downloader.download("golden retriever dog", limit=50, output_dir='train_data')
downloader.download("mountain landscape", limit=50, output_dir='train_data')
downloader.download("sports car", limit=50, output_dir='train_data')