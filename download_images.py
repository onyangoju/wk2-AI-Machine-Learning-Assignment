from icrawler.builtin import BingImageCrawler

def download_images(keyword, target_folder, max_num=50):
    crawler = BingImageCrawler(storage={'root_dir': target_folder})
    crawler.crawl(keyword=keyword, max_num=max_num)

# Create deforested and non-deforested image sets
download_images("deforested satellite image", "dataset/deforested", max_num=50)
download_images("healthy forest satellite image", "dataset/non_deforested", max_num=50)
