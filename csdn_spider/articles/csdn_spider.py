import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import os

class CSDNSpider:
    def __init__(self):
        self.base_url = "https://blog.csdn.net/nav/ai/ml"
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        self.driver = uc.Chrome(options=options)
        self.articles = self.load_existing_articles()  # 初始化时加载已有文章
        self.new_articles_count = 0  # 记录新爬取的文章数量

    def load_existing_articles(self, filename='csdn_articles.json'):
        """加载已有的文章数据"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_articles = json.load(f)
                print(f"已加载现有文章 {len(existing_articles)} 篇")
                return existing_articles
            else:
                print("未找到现有文章文件，将创建新文件")
                return []
        except Exception as e:
            print(f"加载现有文章失败: {e}")
            return []

    def get_existing_ids(self):
        """获取已有文章的ID集合"""
        return set(article['article_id'] for article in self.articles)

    def scroll_page(self, max_scrolls=5):
        """滚动页面加载更多文章"""
        print("开始滚动页面加载更多文章...")
        scrolls = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while scrolls < max_scrolls:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("已到达页面底部")
                break
                
            last_height = new_height
            scrolls += 1
            print(f"已完成第 {scrolls} 次滚动")

    def parse_article_list(self, html):
        """解析文章列表页"""
        soup = BeautifulSoup(html, 'html.parser')
        article_items = soup.select('.article-item-box')
        print(f"找到 {len(article_items)} 篇文章")

        existing_ids = self.get_existing_ids()
        
        for item in article_items:
            try:
                title_element = item.select_one('a.article-title')
                if not title_element:
                    continue
                
                title = title_element.get_text(strip=True)
                link = title_element.get('href', '')
                
                # 提取文章ID
                article_id = link.split('/')[-1] if link else None
                if not article_id or not article_id.isdigit():
                    print(f"无法获取文章ID，跳过: {link}")
                    continue
                
                # 检查文章是否已存在
                if article_id in existing_ids:
                    print(f"文章已存在，跳过: [ID: {article_id}] {title}")
                    continue
                
                # 获取文章预览内容
                content_preview = item.select_one('.article-desc.word-1')
                content = content_preview.get_text(strip=True) if content_preview else ""
                
                # 获取阅读量、点赞数和收藏数
                stats = item.select('span.num')
                read_count = ''
                like_count = ''
                collect_count = ''
                
                if len(stats) >= 3:
                    read_count = stats[0].get_text(strip=True)
                    like_count = stats[1].get_text(strip=True)
                    collect_count = stats[2].get_text(strip=True)
                
                article_data = {
                    'article_id': article_id,
                    'title': title,
                    'link': link,
                    'content_preview': content,
                    'read_count': read_count,
                    'like_count': like_count,
                    'collect_count': collect_count
                }
                
                self.articles.append(article_data)
                self.new_articles_count += 1
                print(f"成功解析新文章: [ID: {article_id}] {title}")
                print(f"阅读量: {read_count}, 点赞数: {like_count}, 收藏数: {collect_count}")
            
            except Exception as e:
                print(f"解析文章失败: {e}")
                continue

    def save_to_json(self, filename='csdn_articles.json'):
        """保存数据到JSON文件"""
        if not self.articles:
            print("没有文章数据可保存")
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=2)
            print(f"成功保存所有文章到 {filename}")
            print(f"本次新增 {self.new_articles_count} 篇文章")
            print(f"当前总共 {len(self.articles)} 篇文章")
        except Exception as e:
            print(f"保存文件失败: {e}")

    def run(self, max_scrolls=50):
        """运行爬虫"""
        try:
            print("开始爬取CSDN文章...")
            print(f"当前已有文章数量: {len(self.articles)}")
            
            self.driver.get(self.base_url)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "article-item-box"))
            )
            
            self.scroll_page(max_scrolls)
            
            html = self.driver.page_source
            self.parse_article_list(html)
            
            if self.new_articles_count > 0:
                self.save_to_json()
            else:
                print("未发现新文章")
                
        except Exception as e:
            print(f"爬虫运行出错: {e}")
        
        finally:
            self.driver.quit()

if __name__ == "__main__":
    spider = CSDNSpider()
    spider.run()