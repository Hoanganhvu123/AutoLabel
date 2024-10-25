import os
import json
import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Union
from langchain_groq import ChatGroq
from langchain.schema import AIMessage
from itertools import cycle
from tqdm import tqdm
import time

class Worker:
    def __init__(self, api_key: str, model: str, session: aiohttp.ClientSession):
        self.api_key = api_key
        self.model = model
        self.session = session
        self.llm = ChatGroq(temperature=0, model=model, groq_api_key=api_key, client=session)
        self.rate_limit = 30  # Requests per minute
        self.last_request_time = 0
        self.request_count = 0

    async def process(self, text: str, context: str, fields: List[Dict]) -> Dict[str, Union[str, Dict[str, str]]]:
        current_time = time.time()
        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
        self.request_count += 1
        
        try:
            prompt = TextAnalyzer._create_core_prompt(text, context, fields)
            result = await self.llm.ainvoke(prompt)
            content = result.content if isinstance(result, AIMessage) else str(result)
            analysis = TextAnalyzer._parse_result(content, fields)
            return {"text": text, "result": analysis}
        except Exception as e:
            print(f"❌ Error with model {self.model}: {str(e)}")
            return {"text": text, "result": {field['name']: field.get('fallback_value', 'error') for field in fields}}

class TextAnalyzer:
    def __init__(self, api_keys: Union[str, List[str]], dataset_path: str, context: str, fields: List[Dict], batch_size: int = 20):
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.dataset_path = dataset_path
        self.context = context
        self.fields = fields
        self.batch_size = batch_size
        self.models = [
            "gemma-7b-it", "gemma2-9b-it", "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview",
            "mixtral-8x7b-32768", "llama-3.2-11b-text-preview",
            "llama-3.2-1b-preview", "llama-3.2-3b-preview",
            "llama-3.2-90b-text-preview"
        ]

    @staticmethod
    def _create_core_prompt(text: str, context: str, fields: List[Dict]) -> str:
        prompt = (
            f"You are an AI expert in data analysis and classification. "
            f"Your task is to analyze the given text based on the provided "
            f"context and fields. Follow these instructions carefully:\n\n"
            f"1. Context: {context}\n\n2. Text to analyze: {text}\n\n"
            f"3. Fields to analyze:\n"
        )
        for field in fields:
            prompt += (f"\n   - {field['name']} ({field['type']}): "
                       f"{field['guidelines']}")
            if 'labels' in field:
                prompt += f"\n     Options: {', '.join([f'{i}: {label}' for i, label in enumerate(field['labels'])])}"

        prompt += (
            "\n\n4. Important instructions:\n"
            "   - Analyze the text thoroughly based on each field's guidelines.\n"
            "   - For each field, provide ONLY the index number of the most appropriate label.\n"
            "   - Do not include any explanations or additional text.\n"
            "   - Format your response as a single integer representing the index of the chosen label.\n\n"
            "5. Response format:\n"
            "   <index_number>\n\n"
            "Now, please analyze the text and provide your response as a single integer."
        )
        return prompt

    @staticmethod
    def _parse_result(content: str, fields: List[Dict]) -> Dict[str, str]:
        try:
            index = int(content.strip())
            field = fields[0]  # Assuming only one field for simplicity
            if 0 <= index < len(field['labels']):
                return {field['name']: field['labels'][index]}
            else:
                return {field['name']: field.get('fallback_value', 'error')}
        except ValueError:
            return {field['name']: field.get('fallback_value', 'error') for field in fields}

    def _create_worker_pool(self, session: aiohttp.ClientSession) -> List[Worker]:
        return [Worker(api_key, model, session) 
                for api_key in self.api_keys 
                for model in self.models]

    async def _process_batches(self, texts: List[str], worker_pool: List[Worker]) -> List[Dict]:
        results = []
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        with tqdm(total=len(texts), desc="⏳ Analyzing texts", unit="text") as pbar:
            for batch in batches:
                tasks = []
                for text, worker in zip(batch, cycle(worker_pool)):
                    tasks.append(self._process_with_retry(text, worker))
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results

    async def _process_with_retry(self, text: str, worker: Worker, max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                return await worker.process(text, self.context, self.fields)
            except Exception as e:
                print(f"❌ Error with worker {worker.model}: {str(e)}. Retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        print(f"❌ All retries failed for text: {text[:50]}...")
        return {"text": text, "result": {field['name']: field.get('fallback_value', 'error') for field in self.fields}}

    def _save_results(self, results: List[Dict[str, Union[str, Dict[str, str]]]], filename: str) -> None:
        print(f"💾 Saving results to {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {filename}.")

    def _save_results_to_csv(self, df: pd.DataFrame, results: List[Dict[str, Union[str, Dict[str, str]]]]) -> None:
        print("💾 Saving results to new.csv...")
        for result in results:
            text = result['text']
            sentiment = result['result']['sentiment']
            df.loc[df['text'] == text, 'classify'] = sentiment
        
        df.to_csv('new.csv', index=False, encoding='utf-8')
        print("✅ Results saved to new.csv.")

    async def run(self):
        print(f"🚀 Starting text analysis task")

        df = pd.read_csv(self.dataset_path)
        print(f"📊 Columns in the dataset: {df.columns.tolist()}")

        if 'text' not in df.columns:
            raise ValueError("The dataset must contain a 'text' column.")

        texts = df['text'].tolist()

        print(f"🔢 Total texts to analyze: {len(texts)}")

        async with aiohttp.ClientSession() as session:
            worker_pool = self._create_worker_pool(session)
            results = await self._process_batches(texts, worker_pool)

        print("💾 Saving results...")
        self._save_results(results, "text_analysis_results.json")
        self._save_results_to_csv(df, results)

        print("🎉 Task completed!")

# Example usage:
if __name__ == "__main__":
    api_keys = ["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"]  # Add more API keys if available
    dataset_path = "E:\\label\\dataset.csv"
    context = "Analyze customer feedback"
    fields = [
        {
            "name": "sentiment",
            "type": "categorical",
            "guidelines": "Classify the sentiment of the text",
            "labels": ["positive", "negative", "neutral"]
        }
    ]

    analyzer = TextAnalyzer(api_keys, dataset_path, context, fields)
    asyncio.run(analyzer.run())
