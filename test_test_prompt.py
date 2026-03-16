import asyncio
import logging
import requests
import json
import re
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
    last_known_price: float


@dataclass
class StepMetrics:
    smape: float
    direction_accuracy: float
    mae: float
    rmse: float
    samples: int


@dataclass
class OverallMetrics:
    total_predictions: int
    avg_smape: float
    avg_direction_accuracy: float
    avg_mae: float
    avg_rmse: float
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

    async def get_response(self, question_text, temperature=0.7, max_tokens=50):
        if self.model == "gpt-oss:120b-cloud" or self.model == "qwen3-vl:235b-cloud":
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.model,
                    prompt=question_text,
                    options={"temperature": temperature, "seed": 42},
                    think="medium"
                )
            )

            i = 1
            response_limit = 5
            while not self._extract_prediction(response.response):
                response = await loop.run_in_executor(
                    None,
                    lambda: ollama.generate(
                        model=self.model,
                        prompt=question_text,
                        options={"temperature": temperature},
                        think="medium"
                    )
                )
                print(f"ATTEMPT {i}: {response.response}")
                i += 1
                if i > response_limit:
                    break

            #print('==============================================================================')
            #print(f'PROMPT: \n{question_text}\n')
            #print('==============================================================================\n\n')
            #print(f"OLLAMA RESPONSE: \n{response.response}\n")
            #print('==============================================================================\n\n')
            return response.response

        elif self.model == "gemma3":
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model='gemma3',
                    messages=[{'role': 'user', 'content': question_text}]
                )
            )
            return response.message.content
        else:
            try:
                headers = await self._prepare_headers()
                data = await self._prepare_request_data(question_text, temperature, max_tokens)

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

    async def test_prompt_on_dataset(self, user_prompt: str, test_dataset: List[Dict], horizon: int = 5) -> Dict[str, any]:
        if not self.session:
            self.session = aiohttp.ClientSession()

        tasks = []
        for test_case in test_dataset:
            task = self._process_test_case(user_prompt, test_case, horizon)
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks, desc="Тестирование")
        results = [r for r in results if r is not None]

        metrics = self._calculate_multistep_metrics(results)

        return {
            "results": results,
            "metrics": metrics,
            "prompt": user_prompt,
            "horizon": horizon
        }

    async def _process_test_case(self, user_prompt: str, test_case: Dict, horizon: int):
        predictions = await self._get_multistep_predictions(user_prompt, test_case, horizon)

        if predictions:
            actual_prices = await self._get_actual_prices(test_case, horizon)
            last_price = self._get_last_context_price(test_case)

            #print('==============================================================================\n')
            #print("REAL PRICES:")
            #print('\n'.join(list(map(str, actual_prices))), '\n')
            #print('==============================================================================\n')

            return MultiStepResult(
                actual_prices=actual_prices,
                predicted_prices=predictions,
                timestamp=test_case['test_date'],
                symbol=test_case['symbol'],
                prompt=user_prompt,
                horizon=horizon,
                last_known_price=last_price
            )
        return None

    async def _get_multistep_predictions(self, user_prompt: str, test_case: Dict, horizon: int) -> List[float]:
        current_context = test_case['context_data']
        full_prompt = self._build_prediction_prompt(user_prompt, current_context, horizon)

        response = await self.get_response(full_prompt)
        predicted_prices = self._extract_prediction(response)

        if not predicted_prices:
            print("Не удалось извлечь предсказание")
            return None

        return predicted_prices[:horizon]

    def _build_prediction_prompt(self, user_prompt: str, context_data: str, horizon: int) -> str:
        return f"""
{user_prompt}

КОНТЕКСТ ДАННЫХ:
{context_data}

ИНСТРУКЦИИ:
- Проанализируй исторические данные
- Предскажи цены закрытия следующих {horizon} свеч
- Каждое числовое значение верни на новой строке в формате: 123.45
- Не добавляй пояснений, текста или символов

ЕЩЕ РАЗ: В ОТВЕТЕ УКАЖИ **ТОЛЬКО {horizon} ЧИСЕЛ - ПРЕДСКАЗАННЫЕ ЦЕНЫ ЗАКРЫТИЯ**

Твой прогноз:
"""

    def _extract_prediction(self, response):
        try:
            lines = response.strip().split('\n')
            predictions = []
            for line in lines:
                line = line.strip()
                if line:
                    line = line.replace(',', '.').strip()
                    try:
                        pred = float(line)
                        predictions.append(pred)
                    except ValueError:
                        continue
            return predictions
        except Exception as e:
            logging.error(f"Ошибка извлечения предсказаний: {e}")
            return []

    def _generate_next_timestamp(self, last_timestamp: str, interval: str) -> str:
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
                next_dt = last_dt + timedelta(hours=1)

            return next_dt.strftime(format_string)
        except Exception as e:
            logging.error(f"Ошибка генерации временной метки: {e}")
            return last_timestamp

    async def _get_actual_prices(self, test_case: Dict, horizon: int) -> List[float]:
        symbol = test_case['symbol']
        interval = test_case['interval']
        start_date = test_case['timestamp']

        actual_prices = []
        prev_date = start_date

        for i in range(horizon):
            new_date = self._generate_next_timestamp(prev_date, interval)

            rows = await self.db_controller.select(
                "candles",
                "symbol = ? AND interval = ? AND datetime = ?",
                (symbol, interval, new_date)
            )

            if rows and len(rows) > 0:
                price = float(rows[0][4])
                actual_prices.append(price)
                prev_date = new_date
            else:
                actual_prices.append(np.nan)
                prev_date = new_date

        return actual_prices

    def _get_last_context_price(self, test_case: Dict) -> float:
        price = float(test_case['context_data'].split('\n')[-1].split(',')[-2])
        return price

    def _calculate_multistep_metrics(self, results: List[MultiStepResult]) -> Optional[OverallMetrics]:
        if not results:
            return None

        H = results[0].horizon
        K = len(results)

        window_mae = []
        window_rmse = []
        window_smape = []
        window_mda = []
        step_metrics = {}
        i = 0

        for r in results:
            y = np.array(r.actual_prices[:H], dtype=float)
            y_hat = np.array(r.predicted_prices[:H], dtype=float)
            last_price = r.last_known_price

            # ===== MAE =====
            mae_i = np.mean(np.abs(y - y_hat))

            # ===== RMSE =====
            rmse_i = np.sqrt(np.mean((y - y_hat) ** 2))

            # ===== sMAPE =====
            denom = np.abs(y) + np.abs(y_hat)
            smape_i = np.mean(
                np.where(denom == 0, 0, 2 * np.abs(y - y_hat) / denom)
            ) * 100

            # ===== MDA =====
            prev_actual = np.concatenate([[last_price], y[:-1]])

            actual_direction = np.sign(y - prev_actual)
            forecast_direction = np.sign(y_hat - prev_actual)

            mda_i = np.mean((actual_direction == forecast_direction).astype(float))

            window_mae.append(mae_i)
            window_rmse.append(rmse_i)
            window_smape.append(smape_i)
            window_mda.append(mda_i)

            step_metrics[f'step_{i}'] = StepMetrics(
                smape=smape_i,
                direction_accuracy=mda_i,
                mae=mae_i,
                rmse=rmse_i,
                samples=999
            )
            i += 1

        return OverallMetrics(
            total_predictions=K * H,
            avg_smape=float(np.mean(window_smape)),
            avg_direction_accuracy=float(np.mean(window_mda)),
            avg_mae=float(np.mean(window_mae)),
            avg_rmse=float(np.mean(window_rmse)),
            step_metrics=step_metrics  # больше не используем
        )

    def generate_report(self, test_results: Dict) -> str:
        metrics = test_results['metrics']

        step_lines = []
        for step_name, sm in metrics.step_metrics.items():
            step_lines.append(
                f"  {step_name}: SMAPE={sm.smape:.2f}% | DA={sm.direction_accuracy:.1%} | "
                f"MAE={sm.mae:.2f} | RMSE={sm.rmse:.2f} | samples={sm.samples}"
            )
        step_report = '\n'.join(step_lines)

        report = f"""
📊 ОТЧЕТ О ТЕСТИРОВАНИИ ПРОМПТА
{'=' * 50}

📈 ОСНОВНЫЕ МЕТРИКИ КАЧЕСТВА:
• Количество прогнозов: {metrics.total_predictions}
• SMAPE: {metrics.avg_smape:.2f}%
• MAE: {metrics.avg_mae:.2f}
• RMSE: {metrics.avg_rmse:.2f}
• Direction Accuracy: {metrics.avg_direction_accuracy:.1%}

📉 МЕТРИКИ ПО ШАГАМ:
{step_report}

📝 ТЕСТИРУЕМЫЙ ПРОМПТ:
{test_results['prompt'][:250]}...

⚙️ ПАРАМЕТРЫ ТЕСТА:
• Горизонт прогнозирования: {test_results['horizon']} свечей
• Количество семплов: {len(test_results['results'])}
• Модель: {getattr(self, 'model', 'YandexGPT')}

{'=' * 50}
"""
        return report


if __name__ == "__main__":
    IAM_TOKEN = ""
    FOLDER_ID = "your_folder_id"
    MODEL_URI = "your_model_uri"
    db_controller = DBController()

    async def test_main():
        async with Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller) as tester:
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