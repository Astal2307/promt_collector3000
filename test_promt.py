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
from tqdm import tqdm
import ollama


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
    def __init__(self, iam_token, folder_id, model_uri, db_controller, model="gpt-oss:120b-cloud"):
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.model_uri = model_uri
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        self.model = model

        self.db_controller = db_controller
        
    def _prepare_headers(self):
        return {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
            "Content-Type": "application/json"
        }
    
    def _prepare_request_data(self, question_text, temperature=0.6, max_tokens=50):
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
            response = ollama.generate(model=self.model, prompt=question_text)

            return response.response
        elif self.model == "gemma3":
            response: ChatResponse = chat(model='gemma3', messages=[
                    {
                        'role': 'user',
                        'content': question_text,
                    },
                ])
            return response.message.content
        else:
            try:
                headers = self._prepare_headers()
                data = self._prepare_request_data(question_text, temperature, max_tokens)
                
                response = await self._make_async_request(headers, data)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["result"]["alternatives"][0]["message"]["text"]
                else:
                    logging.error(f"Ошибка API: {response.status_code} - {response.text}")
                    return f"Ошибка при обращении к API: {response.status_code}"
                    
            except Exception as e:
                logging.error(f"Ошибка при запросе к YandexGPT: {e}")
                return "Извините, произошла ошибка при обработке запроса."
    
    async def _make_async_request(self, headers, data):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: requests.post(self.api_url, headers=headers, json=data, timeout=30)
        )
    
    async def test_prompt_on_dataset(self, user_prompt: str, test_dataset: List[Dict], 
                               horizon: int = 10) -> Dict[str, any]:
        results = []
        
        for test_case in tqdm(test_dataset):
            predictions = await self._get_multistep_predictions(
                user_prompt, test_case, horizon
            )
            
            if predictions:
                actual_prices = self._get_actual_prices(test_case, horizon)
                
                result = MultiStepResult(
                    actual_prices=actual_prices,
                    predicted_prices=predictions,
                    timestamp=test_case['test_date'],
                    symbol=test_case['symbol'],
                    prompt=user_prompt,
                    horizon=horizon
                )
                results.append(result)
        
        metrics = self._calculate_multistep_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'prompt': user_prompt,
            'horizon': horizon
        }

    async def _get_multistep_predictions(self, user_prompt: str, test_case: Dict, 
                                    horizon: int) -> List[float]:
        predictions = []
        current_context = test_case['context_data']
        
        for step in range(horizon):
            full_prompt = self._build_prediction_prompt(user_prompt, current_context)
            
            response = await self.get_response(full_prompt)
            predicted_price = self._extract_prediction(response)
            
            if predicted_price is None:
                logging.warning(f"Не удалось извлечь предсказание на шаге {step}")
                break
                
            predictions.append(predicted_price)
            
            current_context = self._update_context(current_context, predicted_price, step, test_case['interval'])
        
        return predictions

    def _build_prediction_prompt(self, user_prompt: str, context_data: str) -> str:
        return f"""
    {user_prompt}

    КОНТЕКСТ ДАННЫХ:
    {context_data}

    ИНСТРУКЦИИ:
    - Проанализируй исторические данные
    - Предскажи цену закрытия следующей свечи
    - Верни только числовое значение в формате: 123.45
    - Не добавляй пояснений, текста или символов

    Твой прогноз:
    """

    def _extract_prediction(self, response):
        # whaaa
        return float(response)

    def _update_context(self, current_context: str, predicted_price: float, 
                    step: int, interval: str) -> str:
        lines = current_context.strip().split('\n')
        
        last_line = lines[-1]
        last_parts = last_line.split(',')
        
        predicted_date = self._generate_next_timestamp(last_parts[0], interval)
        predicted_open = float(last_parts[4])
        predicted_high = max(predicted_open, predicted_price) * 1.01  # +1%
        predicted_low = min(predicted_open, predicted_price) * 0.99   # -1%
        predicted_volume = float(last_parts[5])
        
        new_candle = (f"{predicted_date},"
                    f"{predicted_open:.6f},{predicted_high:.6f},{predicted_low:.6f},"
                    f"{predicted_price:.6f},{predicted_volume:.2f}")
        
        updated_lines = lines[1:] + [new_candle]
        return '\n'.join(updated_lines)

    def _generate_next_timestamp(self, last_timestamp: str, interval: str) -> str:
        from datetime import datetime, timedelta

        format_string = "%Y-%m-%d %H:%M:%S"
        last_timestamp = datetime.strptime(last_timestamp, format_string)
        try:           
            if interval == "1h":
                return (last_timestamp + timedelta(hours=1)).strftime(format_string)
            elif interval == "1d":
                return (last_timestamp + timedelta(days=1)).strftime(format_string)
            else:
                return (last_timestamp + timedelta(weeks=1)).strftime(format_string)
        except:
            return last_timestamp

    def _get_actual_prices(self, test_case: Dict, horizon: int) -> List[float]:
        symbol = test_case['symbol']
        interval = test_case['interval']
        start_date = test_case['timestamp']
        prev_date = start_date

        actual_prices = []

        for i in range(horizon):
            new_date = self._generate_next_timestamp(prev_date, interval)
            price = float(self.db_controller.select("candles", "symbol = ? AND interval = ? AND datetime = ?", (symbol, interval, new_date))[0][-2])
            actual_prices.append(price)

        return actual_prices

    def _calculate_multistep_metrics(self, results: List[MultiStepResult]) -> OverallMetrics:
        if not results:
            return OverallMetrics(0, 0.0, 0.0, 0.0, {})
        
        step_metrics = {}
        all_mape_errors = []
        all_directions = []
        total_predictions = 0
        
        for step in range(results[0].horizon):
            step_mape_errors = []
            step_mae_errors = []
            step_directions = []
            step_samples = 0
            
            for result in results:
                if step < len(result.predicted_prices) and step < len(result.actual_prices):
                    actual = result.actual_prices[step]
                    predicted = result.predicted_prices[step]
                    
                    # MAE
                    mae_error = abs(actual - predicted)
                    step_mae_errors.append(mae_error)
                    
                    # MAPE
                    if actual != 0:
                        mape_error = mae_error / actual
                        step_mape_errors.append(mape_error)
                        all_mape_errors.append(mape_error)
                    
                    # Direction accuracy
                    if step > 0:
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
        # success_rate = self._calculate_overall_success_rate(results)
        
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
    IAM_TOKEN = 't1.9euelZrOlJGUyYnMnsjGxpSTzciey-3rnpWajI2UyJOPyoyLi4mVm4mPnZjl9PcBOFo2-e86GQ6z3fT3QWZXNvnvOhkOs83n9euelZrMl52UnJzMm8mJnMiRi8zPku_8xeuelZrMl52UnJzMm8mJnMiRi8zPkg.SG8Hnp1M7htfIDeF--G-YRWtbBz5XfEKrVYPE6d0dI-tRVhzhJhlEZPCy-iNitGVBmvcfI1M2MzZ_Q3uLW_LAg'
    FOLDER_ID = "b1g2607c57t2ou6p7c0o"
    MODEL_URI = "gpt://b1g2607c57t2ou6p7c0o/yandexgpt/latest"
    db_controller = DBController()
    tester = Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller, model="YandexGPT")
    results = asyncio.run(tester.test_prompt_on_dataset(
        user_prompt="Твой лучший промпт для прогнозирования цен",
        test_dataset=db_controller.sample_data(num_samples=3),
        horizon=5
    ))

    print(tester.generate_report(results))