import asyncio
import logging
import requests
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from db_controller import DBController
from ollama import chat
from ollama import ChatResponse
from tqdm.asyncio import tqdm_asyncio
import ollama
import aiohttp


@dataclass
class MultiStepResult:
    actual_prices: List[float]
    predicted_prices: List[float]
    timestamp: str
    symbol: str
    prompt: str
    horizon: int
    
@dataclass
class StepMetrics:
    mape: float
    direction_accuracy: float
    samples: int

@dataclass
class OverallMetrics:
    total_predictions: int
    avg_mape: float
    avg_direction_accuracy: float
    step_metrics: Dict[str, StepMetrics]

class Tester:
    def __init__(self, iam_token, folder_id, model_uri, db_controller: DBController, model="gpt-oss:120b-cloud"):
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.model_uri = model_uri
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        self.model = model
        self.db_controller = db_controller
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _prepare_headers(self):
        return {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json"
        }
    
    async def _prepare_request_data(self, question_text, temperature=0.6, max_tokens=50):
        return {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "messages": [
                {
                    "role": "user",
                    "text": question_text
                }
            ]
        }
    
    async def get_response(self, question_text, temperature=0.6, max_tokens=50):
        if self.model == "gpt-oss:120b-cloud":
            # Используем run_in_executor для синхронного вызова ollama
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(model=self.model, prompt=question_text)
            )

            # print(f"ollama response: \n{response.response}\n\n")
            return response.response

        elif self.model == "gemma3":
            # Используем run_in_executor для синхронного вызова ollama.chat
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(model='gemma3', messages=[
                    {'role': 'user', 'content': question_text}
                ])
            )
            return response.message.content
        else:
            try:
                headers = await self._prepare_headers()
                data = await self._prepare_request_data(question_text, temperature, max_tokens)
                
                # Используем aiohttp вместо requests
                async with self.session.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result["result"]["alternatives"][0]["message"]["text"]
                    else:
                        error_text = await response.text()
                        logging.error(f"Ошибка API: {response.status} - {error_text}")
                        return f"Ошибка при обращении к API: {response.status}"
                    
            except Exception as e:
                logging.error(f"Ошибка при запросе к YandexGPT: {e}")
                return "Извините, произошла ошибка при обработке запроса."
    
    async def test_prompt_on_dataset(self, user_prompt: str, test_dataset: List[Dict], 
                               horizon: int = 10) -> Dict[str, any]:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        tasks = []
        for test_case in test_dataset:
            task = self._process_test_case(user_prompt, test_case, horizon)
            tasks.append(task)
        
        results = []
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Тестирование"):
            result = await task
            if result:
                results.append(result)
        
        metrics = self._calculate_multistep_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'prompt': user_prompt,
            'horizon': horizon
        }
    
    async def _process_test_case(self, user_prompt: str, test_case: Dict, horizon: int):
        predictions = await self._get_multistep_predictions(
            user_prompt, test_case, horizon
        )
        
        if predictions:
            actual_prices = await self._get_actual_prices(test_case, horizon)
            
            return MultiStepResult(
                actual_prices=actual_prices,
                predicted_prices=predictions,
                timestamp=test_case['test_date'],
                symbol=test_case['symbol'],
                prompt=user_prompt,
                horizon=horizon
            )
        return None

    async def _get_multistep_predictions(self, user_prompt: str, test_case: Dict, 
                                    horizon: int) -> List[float]:
        current_context = test_case['context_data']
        
        full_prompt = self._build_prediction_prompt(user_prompt, current_context, horizon)
            
        response = await self.get_response(full_prompt)
        predicted_prices = self._extract_prediction(response)
        
        if None in predicted_prices:
            logging.warning(f"Не удалось извлечь предсказание на шаге {predicted_prices.index(None) + 1}")
            return None
        
        return predicted_prices

    def _build_prediction_prompt(self, user_prompt: str, context_data: str, horizon: int) -> str:
        """Синхронный метод для создания промпта"""
        return f"""
{user_prompt}

КОНТЕКСТ ДАННЫХ:
{context_data}

ИНСТРУКЦИИ:
- Проанализируй исторические данные
- Предскажи цены закрытия следующих {horizon} свеч
- Каждое числовое значение верни на новой строке в формате: 123.45
- Не добавляй пояснений, текста или символов

Твой прогноз:
"""

    def _extract_prediction(self, response):
        """Синхронный метод для извлечения предсказаний"""
        try:
            # Разделяем строки и пытаемся преобразовать в числа
            lines = response.strip().split('\n')
            predictions = []
            for line in lines:
                line = line.strip()
                if line:
                    # Убираем возможные лишние символы
                    line = line.replace(',', '.').strip()
                    try:
                        pred = float(line)
                        predictions.append(pred)
                    except ValueError:
                        # Пропускаем строки, которые нельзя преобразовать в число
                        continue
            return predictions
        except Exception as e:
            logging.error(f"Ошибка извлечения предсказаний: {e}")
            return []

    def _generate_next_timestamp(self, last_timestamp: str, interval: str) -> str:
        """Синхронный метод для генерации временной метки"""
        from datetime import datetime, timedelta

        try:
            format_string = "%Y-%m-%d %H:%M:%S"
            last_dt = datetime.strptime(last_timestamp, format_string)
            
            if interval == "1h":
                next_dt = last_dt + timedelta(hours=1)
            elif interval == "1d":
                next_dt = last_dt + timedelta(days=1)
            elif interval == "1w":
                next_dt = last_dt + timedelta(weeks=1)
            else:
                next_dt = last_dt + timedelta(hours=1)  # по умолчанию 1 час
            
            return next_dt.strftime(format_string)
        except Exception as e:
            logging.error(f"Ошибка генерации временной метки: {e}")
            return last_timestamp

    async def _get_actual_prices(self, test_case: Dict, horizon: int) -> List[float]:
        """Асинхронное получение фактических цен"""
        symbol = test_case['symbol']
        interval = test_case['interval']
        start_date = test_case['timestamp']
        
        actual_prices = []
        prev_date = start_date

        for i in range(horizon):
            new_date = self._generate_next_timestamp(prev_date, interval)
            
            # Используем run_in_executor для синхронного вызова select
            loop = asyncio.get_event_loop()
            rows = await self.db_controller.select(
                "candles", 
                "symbol = ? AND interval = ? AND datetime = ?", 
                (symbol, interval, new_date)
            )
            
            if rows and len(rows) > 0:
                price = float(rows[0][4])  # close price (индекс 4 в candles)
                actual_prices.append(price)
                prev_date = new_date
            else:
                # Если данных нет, используем последнюю известную цену или 0
                if actual_prices:
                    actual_prices.append(actual_prices[-1])
                else:
                    actual_prices.append(0.0)

        return actual_prices

    def _calculate_multistep_metrics(self, results: List[MultiStepResult]) -> OverallMetrics:
        """Синхронный метод для расчета метрик"""
        if not results:
            return OverallMetrics(0, 0.0, 0.0, {})
        
        step_metrics = {}
        all_mape_errors = []
        all_directions = []
        total_predictions = 0
        
        horizon = results[0].horizon if results else 0
        
        for step in range(horizon):
            step_mape_errors = []
            step_directions = []
            step_samples = 0
            
            for result in results:
                if (step < len(result.predicted_prices) and 
                    step < len(result.actual_prices) and
                    result.actual_prices[step] != 0):
                    
                    actual = result.actual_prices[step]
                    predicted = result.predicted_prices[step]
                    
                    # MAE
                    mae_error = abs(actual - predicted)
                    
                    # MAPE
                    mape_error = mae_error / actual
                    step_mape_errors.append(mape_error)
                    all_mape_errors.append(mape_error)
                    
                    # Direction accuracy
                    if step > 0 and len(result.actual_prices) > step:
                        actual_dir = 1 if actual > result.actual_prices[step-1] else 0
                        predicted_dir = 1 if predicted > result.predicted_prices[step-1] else 0
                        dir_correct = 1 if actual_dir == predicted_dir else 0
                        step_directions.append(dir_correct)
                        all_directions.append(dir_correct)
                    
                    step_samples += 1
            
            step_metrics[f'step_{step}'] = StepMetrics(
                mape=np.mean(step_mape_errors) * 100 if step_mape_errors else 0.0,
                direction_accuracy=np.mean(step_directions) if step_directions else 0.0,
                samples=step_samples
            )
            
            total_predictions += step_samples
        
        overall_mape = np.mean(all_mape_errors) * 100 if all_mape_errors else 0.0
        overall_direction = np.mean(all_directions) if all_directions else 0.0
        
        return OverallMetrics(
            total_predictions=total_predictions,
            avg_mape=overall_mape,
            avg_direction_accuracy=overall_direction,
            step_metrics=step_metrics
        )

    def generate_report(self, test_results: Dict) -> str:
        """Генерирует отчет о тестировании промпта"""
        metrics = test_results['metrics']
        
        report = f"""
📊 ОТЧЕТ О ТЕСТИРОВАНИИ ПРОМПТА
{'='*50}

📈 ОСНОВНЫЕ МЕТРИКИ КАЧЕСТВА:
• Количество прогнозов: {metrics.total_predictions}
• MAPE: {metrics.avg_mape:.2f}%
• Accuracy направления: {metrics.avg_direction_accuracy:.1%}

📝 ТЕСТИРУЕМЫЙ ПРОМПТ:
{test_results['prompt'][:250]}...

⚙️ ПАРАМЕТРЫ ТЕСТА:
• Горизонт прогнозирования: {test_results['horizon']} свечей
• Количество семплов: {len(test_results['results'])}
• Модель: {getattr(self, 'model', 'YandexGPT')}

{'='*50}
"""
        return report


if __name__ == "__main__":
    IAM_TOKEN = "" #get_iam_token(OAUTH_TOKEN)
    FOLDER_ID = "your_folder_id"
    MODEL_URI = "your_model_uri"
    db_controller = DBController()
    
    async def test_main():
        async with Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller) as tester:
            # Получаем данные через run_in_executor
            loop = asyncio.get_event_loop()
            test_dataset = await loop.run_in_executor(
                None,
                lambda: db_controller.sample_data(num_samples=10)
            )
            
            results = await tester.test_prompt_on_dataset(
                user_prompt="Сделай все по инструкции",
                test_dataset=test_dataset,
                horizon=5
            )
            
            print(tester.generate_report(results))
    
    asyncio.run(test_main())