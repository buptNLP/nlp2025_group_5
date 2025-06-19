import json
import os
import time
import random
from typing import List, Dict
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class QAGenerator:
    def __init__(self, api_key: str, base_url: str = "Please replace with actual API URL"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.lock = threading.Lock()
        
    def load_json_data(self, json_file: str) -> List[Dict]:
        """加载JSON文章数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_txt_content(self, txt_file: str) -> str:
        """加载TXT文章内容"""
        if not os.path.exists(txt_file):
            return ""
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 限制内容长度，避免token过多
                return content[:3000] if len(content) > 3000 else content
        except:
            return ""
    
    def create_qa_prompt(self, title: str, preview: str, content: str) -> str:
        """创建QA生成提示词"""
        prompt = f"""基于以下机器学习技术文章，生成3个高质量的问答对。要求：
1. 问题要自然、多样化，覆盖不同角度
2. 答案要准确、简洁，基于文章内容
3. 重点关注机器学习术语和概念
4. 格式严格按照JSON数组返回

文章标题：{title}
文章摘要：{preview}
文章内容：{content[:1500]}...

请生成格式如下的JSON：
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}},
  {{"question": "问题3", "answer": "答案3"}}
]"""
        return prompt
    
    def call_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用API生成QA对"""
        payload = {
            "model": "Please replace with actual model name",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"API错误 {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"请求失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    def parse_qa_response(self, response: str) -> List[Dict]:
        """解析API返回的QA对"""
        try:
            # 尝试直接解析JSON
            qa_pairs = json.loads(response)
            if isinstance(qa_pairs, list):
                return qa_pairs
        except:
            pass
        
        # 如果直接解析失败，尝试提取JSON部分
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                qa_pairs = json.loads(json_str)
                return qa_pairs
        except:
            pass
        
        print(f"解析失败的响应: {response[:200]}...")
        return []
    
    def process_single_article(self, article: Dict, txt_dir: str) -> List[Dict]:
        """处理单篇文章生成QA对"""
        article_id = article.get('article_id', '')
        title = article.get('title', '')
        preview = article.get('content_preview', '')
        
        if not title or not preview:
            return []
        
        # 加载对应的TXT文件
        txt_file = os.path.join(txt_dir, f"{article_id}.txt")
        content = self.load_txt_content(txt_file)
        
        # 如果没有详细内容，只用预览
        if not content:
            content = preview
        
        # 生成提示词
        prompt = self.create_qa_prompt(title, preview, content)
        
        # 调用API
        response = self.call_api(prompt)
        if not response:
            return []
        
        # 解析QA对
        qa_pairs = self.parse_qa_response(response)
        
        # 添加元数据
        for qa in qa_pairs:
            qa['source_article_id'] = article_id
            qa['source_title'] = title
            qa['source_link'] = article.get('link', '')
        
        return qa_pairs
    
    def generate_qa_batch(self, json_file: str, txt_dir: str, 
                         output_file: str, max_articles: int = None,
                         max_workers: int = 3) -> None:
        """批量生成QA对"""
        
        # 加载文章数据
        articles = self.load_json_data(json_file)
        if max_articles:
            articles = articles[:max_articles]
        
        print(f"开始处理 {len(articles)} 篇文章")
        
        all_qa_pairs = []
        processed_count = 0
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_article = {
                executor.submit(self.process_single_article, article, txt_dir): article 
                for article in articles
            }
            
            # 处理结果
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    qa_pairs = future.result()
                    if qa_pairs:
                        all_qa_pairs.extend(qa_pairs)
                        print(f"✓ 文章 {article.get('article_id', 'unknown')} 生成 {len(qa_pairs)} 个QA对")
                    else:
                        print(f"✗ 文章 {article.get('article_id', 'unknown')} 生成失败")
                        
                except Exception as e:
                    print(f"✗ 处理文章 {article.get('article_id', 'unknown')} 时出错: {e}")
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"进度: {processed_count}/{len(articles)}")
                
                # 控制请求频率
                time.sleep(0.5)
        
        # 保存结果
        print(f"\n总共生成 {len(all_qa_pairs)} 个QA对")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"QA对已保存到: {output_file}")
    
    def filter_qa_pairs(self, qa_file: str, output_file: str, min_length: int = 10):
        """过滤和清理QA对"""
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        filtered_pairs = []
        for qa in qa_pairs:
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            
            # 基本过滤条件
            if (len(question) >= min_length and 
                len(answer) >= min_length and
                '?' in question or '什么' in question or '如何' in question):
                filtered_pairs.append(qa)
        
        print(f"过滤前: {len(qa_pairs)} 个QA对")
        print(f"过滤后: {len(filtered_pairs)} 个QA对")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_pairs, f, ensure_ascii=False, indent=2)

def main():
    # 配置参数
    API_KEY = ""  # 请替换为实际的API KEY
    JSON_FILE = "csdn_articles.json"
    TXT_DIR = "./articles"  # TXT文件所在目录
    OUTPUT_FILE = "generated_qa_pairs.json"
    FILTERED_OUTPUT = "filtered_qa_pairs.json"
    
    # 初始化生成器
    generator = QAGenerator(API_KEY)
    
    # 生成QA对（可以限制数量）
    generator.generate_qa_batch(
        json_file=JSON_FILE,
        txt_dir=TXT_DIR,
        output_file=OUTPUT_FILE,
        max_articles=827,  # 先处理50篇文章测试
        max_workers=2     # 并发数控制
    )
    
    # 过滤清理
    generator.filter_qa_pairs(OUTPUT_FILE, FILTERED_OUTPUT)
    
    print("QA对生成完成！")

if __name__ == "__main__":
    main()