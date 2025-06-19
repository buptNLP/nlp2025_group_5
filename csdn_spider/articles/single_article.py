import json
import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

class CSDNContentSpider:
    def __init__(self):
        # 设置Chrome选项
        options = uc.ChromeOptions()
        options.add_argument('--headless')  # 无头模式
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # 添加性能优化参数
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        self.driver = uc.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)  # 恢复到10秒，但添加重试机制
        
        # 创建articles目录
        if not os.path.exists('articles'):
            os.makedirs('articles')

    def load_articles(self, json_file='csdn_articles.json'):
        """加载文章数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def wait_for_content(self, max_retries=3):
        """等待内容加载，带重试机制"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 首先等待页面基本元素加载
                self.wait.until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # 检查内容区域
                content_element = self.driver.find_element(By.ID, "content_views")
                
                # 如果内容区域存在但为空，等待一段时间
                if not content_element.text.strip():
                    time.sleep(2)
                    content_element = self.driver.find_element(By.ID, "content_views")
                
                if content_element.text.strip():
                    return True
                    
                retry_count += 1
                time.sleep(2)
            except Exception as e:
                print(f"等待内容加载失败，尝试重试 {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(2)
        
        return False

    def check_and_handle_read_more(self, max_retries=2):
        """检查并处理阅读全文按钮，带重试机制"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                # 检查是否存在阅读全文按钮
                read_more_elements = self.driver.find_elements(By.CLASS_NAME, "btn-readmore")
                
                if not read_more_elements:
                    return True
                    
                for btn in read_more_elements:
                    if btn.is_displayed():
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(1)  # 短暂等待
                        return True
                        
                return True  # 如果没有可见的按钮，也返回成功
                
            except Exception as e:
                print(f"处理阅读全文按钮失败，尝试重试 {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(1)
        
        return False

    def extract_article_content(self, html):
        """提取文章内容"""
        soup = BeautifulSoup(html, 'html.parser')
        content_div = soup.find('div', id='content_views')
        
        if not content_div:
            return None

        content_parts = []
        
        # 处理所有文本内容
        for element in content_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre']):
            text = element.get_text(strip=True)
            if text:
                content_parts.append(text)

        return '\n\n'.join(content_parts) if content_parts else None

    def save_content(self, article_id, content):
        """保存文章内容到文件"""
        filename = os.path.join('articles', f'{article_id}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已保存文章 {article_id} 的内容")

    def crawl_article_content(self, article):
        """爬取单篇文章的内容"""
        try:
            article_id = article['article_id']
            link = article['link']
            
            # 检查文件是否已存在
            if os.path.exists(f'articles/{article_id}.txt'):
                print(f"文章 {article_id} 已存在，跳过")
                return True
            
            print(f"\n正在爬取文章 {article_id}")
            print(f"文章链接: {link}")
            
            self.driver.get(link)
            
            # 等待内容加载
            if not self.wait_for_content():
                print(f"文章 {article_id} 内容加载失败，跳过")
                return False
            
            # 处理阅读全文按钮
            if not self.check_and_handle_read_more():
                print(f"文章 {article_id} 处理阅读全文按钮失败")
            
            # 再次等待内容加载（点击按钮后）
            time.sleep(1)
            
            # 获取并保存内容
            content = self.extract_article_content(self.driver.page_source)
            if content:
                self.save_content(article_id, content)
                return True
            else:
                print(f"无法提取文章 {article_id} 的内容")
                return False

        except Exception as e:
            print(f"爬取文章 {article_id} 时出错: {e}")
            return False

    def run(self):
        """运行爬虫"""
        try:
            articles = self.load_articles()
            total_articles = len(articles)
            print(f"共找到 {total_articles} 篇文章待爬取")
            
            failed_articles = []  # 记录失败的文章

            for i, article in enumerate(articles, 1):
                print(f"\n正在处理第 {i}/{total_articles} 篇文章")
                success = self.crawl_article_content(article)
                
                if not success:
                    failed_articles.append(article)
                
                # 随机延时，但控制在较小范围内
                time.sleep(random.uniform(1, 3))

            # 重试失败的文章
            if failed_articles:
                print(f"\n开始重试失败的文章，共 {len(failed_articles)} 篇")
                for article in failed_articles:
                    print(f"\n重试文章 {article['article_id']}")
                    self.crawl_article_content(article)
                    time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"运行出错: {e}")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    spider = CSDNContentSpider()
    spider.run()
