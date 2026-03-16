import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import ReplyKeyboardMarkup, KeyboardButton
from telegram.error import NetworkError
import sqlite3
import os
from test_test_prompt import Tester
import requests
from db_controller import DBController
import asyncio
import aiosqlite
from datetime import datetime
from collections import deque
import uuid
from checker import Checker
import warnings
import json
from html import escape

warnings.filterwarnings("ignore")

"""logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)"""

def get_iam_token(oauth_token):
    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    headers = {"Content-Type": "application/json"}
    data = {"yandexPassportOauthToken": oauth_token}
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()["iamToken"]

with open('private_info.json', 'r') as f:
    private_info = json.load(f)

OAUTH_TOKEN = private_info['OAUTH_TOKEN']
IAM_TOKEN = "" #get_iam_token(OAUTH_TOKEN)
FOLDER_ID = private_info['FOLDER_ID']
MODEL_URI = private_info['MODEL_URI']
BOT_TOKEN = private_info['BOT_TOKEN']
MAX_ATTEMPTS = 1000
NUM_WINDOWS = 3

request_queue = asyncio.Queue(maxsize=100)
# checker = Checker()

test_prompt = """
Ты — генератор сценариев движения рынка для анализа рисков.

Задача:
   - Создай три разных, но правдоподобных варианта (сценария) будущего движения цены актива на основе его исторических данных.

Этапы:
   - Проанализируй данные: определи, как обычно ведет себя цена (ее волатильность), и найди ключевые уровни поддержки и сопротивления.

   - Придумай три сценария:
        1) Боковой: цена застревает в узком диапазоне.
        2) Рост: цена сильно растет, пробивая сопротивление.
        3) Спад: цена сильно падает, пробивая поддержку.

Выбери один: выбери тот сценарий, который сильнее всего похож на прошлое поведение актива, но при этом является наименее очевидным и линейным.
"""

test_prompt_metrics = {'smape': 2.269,
                       'direction_accuracy': 41.7,
                       'mae': 1238.465,
                       'rmse': 1722.624}

db_controller = DBController()

async def init_db():
    async with aiosqlite.connect('user_prompts.db') as conn:
        cols = ['user_id INTEGER', 'username TEXT', 'prompt TEXT', 'smape FLOAT', 'da FLOAT', 'mae FLOAT', 'rmse FLOAT']
        for idx in range(1, NUM_WINDOWS + 1):
            cols.extend([f'step_{idx}_smape FLOAT', f'step_{idx}_da FLOAT', f'step_{idx}_mae FLOAT', f'step_{idx}_rmse FLOAT'])

        for idx in range(1, NUM_WINDOWS + 1):
            cols.append(f'step_{idx}_preds TEXT')

        for col in cols:
            try:
                await conn.execute(f"ALTER TABLE prompts ADD COLUMN {col};")
            except Exception:
                continue

        await conn.execute(f'''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                {', '.join(cols)}
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_attempts (
                user_id INTEGER PRIMARY KEY,
                attempts_used INTEGER DEFAULT 0,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await conn.commit()


async def save_prompt(user_id, username, prompt, overall_metrics, window_metrics, model_predictions):
    async with aiosqlite.connect('user_prompts.db') as conn:
        cols = ['user_id', 'username', 'prompt', 'smape', 'da', 'mae', 'rmse']
        for idx in range(1, NUM_WINDOWS + 1):
            cols.extend([f'step_{idx}_smape', f'step_{idx}_da', f'step_{idx}_mae', f'step_{idx}_rmse'])
        
        for idx in range(1, NUM_WINDOWS + 1):
            cols.append(f'step_{idx}_preds')

        values = list((user_id, username, prompt, 
                        round(overall_metrics.avg_smape, 3), 
                        round(overall_metrics.avg_direction_accuracy, 3),
                        round(overall_metrics.avg_mae, 3),
                        round(overall_metrics.avg_rmse, 3)))
        for idx in range(NUM_WINDOWS):
            step_metrics = window_metrics[f'step_{idx}']
            values.extend([
                round(step_metrics.smape, 3),
                round(step_metrics.direction_accuracy, 3),
                round(step_metrics.mae, 3),
                round(step_metrics.rmse, 3)
            ])

        for idx in range(NUM_WINDOWS):
            values.append(json.dumps(list(map(lambda x: round(x, 3), model_predictions[idx].predicted_prices))))

        await conn.execute(
            f"INSERT INTO prompts ({', '.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
            tuple(values)
        )
        
        await conn.execute('''
            INSERT OR REPLACE INTO user_attempts (user_id, attempts_used, last_activity)
            VALUES (?, COALESCE((SELECT attempts_used FROM user_attempts WHERE user_id = ?) + 1, 1), ?)
        ''', (user_id, user_id, datetime.now()))
        
        await conn.commit()


async def get_used_attempts(user_id):
    async with aiosqlite.connect('user_prompts.db') as conn:
        cursor = await conn.execute(
            'SELECT attempts_used FROM user_attempts WHERE user_id = ?', 
            (user_id,)
        )
        result = await cursor.fetchone()
        return result[0] if result else 0


async def get_top_users_by_smape(limit=5):
    async with aiosqlite.connect('user_prompts.db') as conn:
        query = '''
            WITH ranked AS (
                SELECT
                    user_id,
                    username,
                    smape,
                    da,
                    mae,
                    rmse,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY smape ASC, timestamp DESC
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY user_id) AS attempts,
                    MAX(timestamp) OVER (PARTITION BY user_id) AS last_attempt
                FROM prompts
                WHERE username IS NOT NULL
                  AND smape IS NOT NULL
            )
            SELECT
                username,
                smape AS best_smape,
                da,
                mae,
                rmse,
                attempts,
                last_attempt
            FROM ranked
            WHERE rn = 1
            ORDER BY best_smape ASC
        '''
        if limit:
            query += ' LIMIT ?'
            cursor = await conn.execute(query, (limit,))
        else:
            cursor = await conn.execute(query)

        return await cursor.fetchall()

async def get_top_users_by_da(limit=5):
    async with aiosqlite.connect('user_prompts.db') as conn:
        query = '''
            WITH ranked AS (
                SELECT
                    user_id,
                    username,
                    da,
                    smape,
                    mae,
                    rmse,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY da DESC, timestamp DESC
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY user_id) AS attempts,
                    MAX(timestamp) OVER (PARTITION BY user_id) AS last_attempt
                FROM prompts
                WHERE username IS NOT NULL
                  AND da IS NOT NULL
            )
            SELECT
                username,
                da AS best_da,
                smape,
                mae,
                rmse,
                attempts,
                last_attempt
            FROM ranked
            WHERE rn = 1
            ORDER BY best_da DESC
        '''
        if limit:
            query += ' LIMIT ?'
            cursor = await conn.execute(query, (limit,))
        else:
            cursor = await conn.execute(query)

        return await cursor.fetchall()

async def get_top_users_by_mae(limit=5):
    async with aiosqlite.connect('user_prompts.db') as conn:
        query = '''
            WITH ranked AS (
                SELECT
                    user_id,
                    username,
                    mae,
                    smape,
                    da,
                    rmse,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY mae ASC, timestamp DESC
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY user_id) AS attempts,
                    MAX(timestamp) OVER (PARTITION BY user_id) AS last_attempt
                FROM prompts
                WHERE username IS NOT NULL
                  AND mae IS NOT NULL
            )
            SELECT
                username,
                mae AS best_mae,
                smape,
                da,
                rmse,
                attempts,
                last_attempt
            FROM ranked
            WHERE rn = 1
            ORDER BY best_mae ASC
        '''
        if limit:
            query += ' LIMIT ?'
            cursor = await conn.execute(query, (limit,))
        else:
            cursor = await conn.execute(query)

        return await cursor.fetchall()


async def get_top_users_by_rmse(limit=5):
    async with aiosqlite.connect('user_prompts.db') as conn:
        query = '''
            WITH ranked AS (
                SELECT
                    user_id,
                    username,
                    rmse,
                    smape,
                    da,
                    mae,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY rmse ASC, timestamp DESC
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY user_id) AS attempts,
                    MAX(timestamp) OVER (PARTITION BY user_id) AS last_attempt
                FROM prompts
                WHERE username IS NOT NULL
                  AND rmse IS NOT NULL
            )
            SELECT
                username,
                rmse AS best_rmse,
                smape,
                da,
                mae,
                attempts,
                last_attempt
            FROM ranked
            WHERE rn = 1
            ORDER BY best_rmse ASC
        '''
        if limit:
            query += ' LIMIT ?'
            cursor = await conn.execute(query, (limit,))
        else:
            cursor = await conn.execute(query)

        return await cursor.fetchall()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = await get_used_attempts(user.id)
    remaining_attempts = MAX_ATTEMPTS - used_attempts
    
    intro_steps = [
        f"""
<b>Привет, {user.first_name}.</b>

Этот бот позволяет тестировать промпты, которые прогнозируют цену закрытия на финансовых рынках.

<b>Цель:</b>  
Создать промпт, который максимально точно предсказывает цены закрытия следующих <b>5 свечей</b>.

<b>Данные:</b>  
• Рынок: наиболее ликвидные криптовалюты  
• Период наблюдений: с 2017 года  
• Для монет, появившихся позже, данные берутся с момента выхода на рынок  
• В выборке присутствуют как периоды роста, так и падения цены

<b>Как это работает:</b>
1. Ты отправляешь промпт  
2. Он тестируется на исторических данных  
3. Ты получаешь метрики точности прогноза

<b>Важно:</b>  
Промпт должен вернуть <b>только 5 чисел</b> — цены закрытия, без текста и пояснений.

<b>Формат данных:</b>  
timestamp, open, high, low, close, volume
<b>Глубина истории:</b> последние 100 свечей

<b>Попыток осталось:</b> {remaining_attempts} из {MAX_ATTEMPTS}

Отправь промпт одним сообщением.
"""
    ]
    
    for step in intro_steps:
        await update.message.reply_text(step, parse_mode='HTML')
        await asyncio.sleep(0.5)
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "👇 Выберите действие:",
        reply_markup=reply_markup
    )


async def show_guide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    guide_text = """
<b>Как работает бот</b>

<b>Задача:</b>  
Научиться формулировать промпты, которые дают точные прогнозы.

<b>Процесс:</b>
• Ты пишешь инструкцию для модели  
• Промпт тестируется на исторических данных  
• Результат оценивается метриками

<b>Ожидаемый вывод модели:</b>
• Цены закрытия следующих 5 свечей  
• Только числа в формате: 123.45

<b>Метрики качества:</b>
• <b>SMAPE</b> — средняя процентная ошибка  
• <b>Direction Accuracy</b> — точность направления (рост / падение)  
• <b>MAE</b> — средняя абсолютная ошибка  
• <b>RMSE</b> — ошибка с усиленным штрафом за крупные промахи

Чем точнее логика анализа в промпте, тем выше результат.
"""
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        guide_text,
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def show_example_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    example_explanation = """
<b>Пример промпта</b>

Это демонстрационный промпт от разработчиков.  
Он показывает один из возможных подходов к прогнозированию.

Промпт предлагает модели:
• анализировать историческое движение цены  
• учитывать уровни поддержки и сопротивления  
• оценивать волатильность

Далее модель:
• рассматривает несколько сценариев  
• выбирает наиболее обоснованный  
• формирует прогноз цен закрытия

Это не эталон — его можно и нужно улучшать.
"""
    
    await update.message.reply_text(
        example_explanation,
        parse_mode='HTML'
    )
    
    await asyncio.sleep(1)
    
    await update.message.reply_text(
        f"<pre>{test_prompt}\n</pre>",
        parse_mode='HTML'
    )

    keyboard = [
        [KeyboardButton("📊 Посмотреть метрики примера")],
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "Вы можете взять этот промпт в качестве основы для создания своего",
        reply_markup=reply_markup
    )


async def show_example_metrics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    metrics_text = f"""
<b>Результаты примерного промпта</b>

Метрики, полученные при тестировании демонстрационного промпта:

<b>SMAPE:</b> {test_prompt_metrics['smape']}%  
Средняя процентная ошибка прогноза

<b>Direction Accuracy:</b> {test_prompt_metrics['direction_accuracy']}%  
Доля правильно предсказанных направлений движения

<b>MAE:</b> {test_prompt_metrics['mae']}  
Среднее отклонение прогноза от фактической цены

<b>RMSE:</b> {test_prompt_metrics['rmse']}  
Метрика, сильнее штрафующая за крупные ошибки
 
Цель — улучшить эти значения.
"""
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        metrics_text,
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def prompt_for_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = await get_used_attempts(user.id)
    remaining_attempts = MAX_ATTEMPTS - used_attempts
    
    prompt_guide = f"""
<b>Готов протестировать промпт?</b>

<b>Осталось попыток:</b> {remaining_attempts}

Отправь инструкцию для модели одним сообщением.

Пример:
«Проанализируй исторические данные и спрогнозируй цены закрытия следующих 5 свечей.»

Чёткая и логичная формулировка обычно даёт лучший результат.
"""
    
    await update.message.reply_text(
        prompt_guide,
        parse_mode='HTML'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text
    
    if message_text == "📝 Как это работает?":
        await show_guide(update, context)
        return
    elif message_text == "✨ Пример промпта":
        await show_example_prompt(update, context)
        return
    elif message_text == "📊 Посмотреть метрики примера":
        await show_example_metrics(update, context)
        return
    elif message_text == "🚀 Тестировать промпт":
        await prompt_for_prompt(update, context)
        return
    elif message_text == "🚀 Отправить свой промпт":
        await prompt_for_prompt(update, context)
        return
    elif message_text == "🏆 Топ промптов":
        await show_top_prompts(update, context)
        return
    elif message_text == "📊 Статистика":
        await stats(update, context)
        return
    elif message_text == "❓ Помощь":
        await help_command(update, context)
        return
    
    used_attempts = await get_used_attempts(user.id)
    
    if used_attempts >= MAX_ATTEMPTS:
        await update.message.reply_text(
            "❌ <b>Попытки закончились</b>\n\n"
            f"Вы использовали все {MAX_ATTEMPTS} попыток. "
            "Вы можете посмотреть топ результатов и статистику!",
            parse_mode='HTML'
        )
        return

    keyboard = [
        [KeyboardButton("✅ Да, тестировать"), KeyboardButton("✏️ Переписать")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    
    confirmation_msg = await update.message.reply_text(
        f"<b>Ваш промпт:</b>\n\n{escape(message_text[:200])}...\n\n"
        "<b>Тестировать этот промпт?</b>",
        reply_markup=reply_markup,
        parse_mode='HTML'
    )
    
    context.user_data['pending_prompt'] = message_text
    context.user_data['confirmation_msg_id'] = confirmation_msg.message_id


async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text
    
    if message_text == "✏️ Переписать":
        await update.message.reply_text(
            "✏️ <b>Хорошо! Отправьте исправленный промпт:</b>",
            parse_mode='HTML'
        )
        return
    
    pending_prompt = context.user_data.get('pending_prompt')
    if not pending_prompt:
        await update.message.reply_text(
            "⚠️ <b>Промпт не найден. Пожалуйста, отправьте его заново.</b>",
            parse_mode='HTML'
        )
        return
    
    used_attempts = await get_used_attempts(user.id)
    
    keyboard = [[KeyboardButton("⏳ Тестируется...")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await add_to_queue(update, context, user, pending_prompt, used_attempts)

async def add_to_queue(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                user, message_text: str, used_attempts: int):
    position = request_queue.qsize() + 1
    await update.message.reply_text(f"✅ Запрос принят. Позиция в очереди: {position}")

    print("[+] Added to queue")

    await request_queue.put((update, context, user, message_text, used_attempts))

async def create_worker():
    while True:
        try:
            update, context, user, message_text, used_attempts = await request_queue.get()

            await process_prompt_testing(update, context, user, message_text, used_attempts)

        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            request_queue.task_done()
            #???

async def process_prompt_testing(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                user, message_text: str, used_attempts: int):
    # checked, similarity = checker.check(message_text)
    checked, similarity = True, 1.0
    if not checked:
        print(f"Checker banned message: {message_text}, SIM: {similarity}")
        await update.message.reply_text(
            text="❌ <b>Ошибка тестирования</b>\n\nСистема определила ваш промпт как несодержательный. Попробуйте другой промпт.",
            parse_mode="html"
        )
        return
    try:
        test_dataset = await db_controller.sample_data(symbol="BTCUSDT", interval="1d", num_samples=NUM_WINDOWS)
        #print([test_dataset[i]['actual_price'] for i in range(len(test_dataset))])
        
        async with Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller) as tester:
            results = await tester.test_prompt_on_dataset(
                user_prompt=message_text,
                test_dataset=test_dataset,
                horizon=5
            )

            metrics = results['metrics']
            if not metrics:
                await update.message.reply_text(
                    text="❌ <b>Ошибка тестирования</b>\n\nНе удалось получить метрики. Попробуйте другой промпт.",
                    parse_mode="html"
                )
                return
                
            smape = metrics.avg_smape
            direction_accuracy = metrics.avg_direction_accuracy
            mae = metrics.avg_mae
            rmse = metrics.avg_rmse

            window_metrics = metrics.step_metrics

            model_predictions = results['results']

            #print(model_predictions)

            await save_prompt(user.id, user.username, message_text, metrics, window_metrics, model_predictions)
            
            remaining_attempts = MAX_ATTEMPTS - (used_attempts + 1)
            
            keyboard = [
                [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
                [KeyboardButton("📝 Как это работает?")],
                [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
                [KeyboardButton("🚀 Тестировать промпт")],
            ]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            
            response_text = f"""<b>Промпт протестирован</b>

<b>Метрики:</b>
• <b>SMAPE</b>: {smape:.2f}%  
• <b>Direction Accuracy</b>: {direction_accuracy:.1%}  
• <b>MAE</b>: {mae:.2f}  
• <b>RMSE</b>: {rmse:.2f}

<b>Пояснения:</b>
• SMAPE — средняя процентная ошибка  
• MAE — среднее абсолютное отклонение  
• RMSE — усиленный штраф за крупные ошибки  
• Direction Accuracy — точность направления движения
"""
            progress_msg = await update.message.reply_text(
                response_text,
                reply_markup=reply_markup,
                parse_mode="HTML"
            )
            
    except Exception as e:
        print(f"Ошибка тестирования промпта: {e}")
        
        keyboard = [
            [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
            [KeyboardButton("📝 Как это работает?")],
            [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
            [KeyboardButton("🚀 Тестировать промпт")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=progress_msg.message_id,
            text=(
                "❌ <b>Произошла ошибка</b>\n\n"
                "Не удалось протестировать промпт. Возможно:\n"
                "• Слишком длинный промпт\n"
                "• Проблемы с нейросетью\n"
                "• Неверный формат данных\n\n"
                "Попробуйте упростить промпт или отправить заново."
            ),
            reply_markup=reply_markup,
            parse_mode='HTML'
        )


async def show_top_prompts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_name = user.username if user.username else "Аноним"

    top_smape = await get_top_users_by_smape(limit=5)
    top_da = await get_top_users_by_da(limit=5)
    top_mae = await get_top_users_by_mae(limit=5)
    top_rmse = await get_top_users_by_rmse(limit=5)

    all_smape = await get_top_users_by_smape(limit=None)
    all_da = await get_top_users_by_da(limit=None)
    all_mae = await get_top_users_by_mae(limit=None)
    all_rmse = await get_top_users_by_rmse(limit=None)

    # print(all_da)
    # print(f"DA: {user_da}")
    
    if not all_smape:
        await update.message.reply_text(
            "📭 <b>Пока нет результатов</b>\n\n"
            "Будьте первым, кто протестирует промпт!",
            parse_mode='HTML'
        )
        return
    
    user_scores = {}
    
    def add_scores(leaderboard, metric_name):
        for position, lst in enumerate(leaderboard, 1):
            for row in lst:
                username = row[0]
                if username not in user_scores:
                    user_scores[username] = {
                        'username': username,
                        'scores': position,
                        'places': {metric_name: position}
                    }
                else:
                    user_scores[username]['scores'] += (position)  # 1 за 1-е место, 2 за 2-е и т.д.
                    user_scores[username]['places'][metric_name] = position
    
    def preprocess_total(leaders, less):
        metric_add = [[leaders[0]]]
        for row in leaders[1:]:
            if less:
                if row[1] > metric_add[-1][0][1]:
                    metric_add.append([row])
                else:
                    metric_add[-1].append(row)
            else:
                if row[1] < metric_add[-1][0][1]:
                    metric_add.append([row])
                else:
                    metric_add[-1].append(row)
        return metric_add
    
    # print(top_smape)
    
    add_smape = preprocess_total(all_smape, True)
    add_da = preprocess_total(all_da, False)
    add_mae = preprocess_total(all_mae, True)
    add_rmse = preprocess_total(all_rmse, True)

    add_scores(add_smape, 'smape')
    add_scores(add_da, 'da')
    add_scores(add_mae, 'mae')
    add_scores(add_rmse, 'rmse')

    print(all_smape, all_da, all_mae, all_rmse, sep='\n')

    def get_user_pos(leaderbord, username):
        for i, lsts in enumerate(leaderbord, 1):
            for objs in lsts:
                if objs[0] == username:
                    return i, objs[1], objs[5]
        return None, None, None

    user_smape_pos, user_smape, user_attempts = get_user_pos(add_smape, user_name)
    user_da_pos, user_da, _ = get_user_pos(add_da, user_name)
    user_mae_pos, user_mae, _ = get_user_pos(add_mae, user_name)
    user_rmse_pos, user_rmse, _ = get_user_pos(add_rmse, user_name)

    def get_user_overall_pos(overall_leaderbord, username):
        for pos, info in enumerate(overall_leaderbord, 1):
            if info['username'] == username:
                return pos
        return None
     
    overall_leaderboard = sorted(
        user_scores.values(),
        key=lambda x: x['scores'],
        # reverse=True
    )
    print(overall_leaderboard)
    # {'username': 'Artyom2307', 'scores': 4, 'places': {'smape': 1, 'da': 1, 'mae': 1, 'rmse': 1}}

    user_overall_pos = get_user_overall_pos(overall_leaderboard, user_name)
        
    # 1. SMAPE
    smape_text = "<b>Лидерборд — SMAPE</b>\n\n"
    smape_text += "Симметричная процентная ошибка.\n"
    smape_text += "Меньше — лучше.\n\n"

    users_was_smape = []
    
    for i, (username, best_smape, da, mae, rmse, attempts, last_attempt) in enumerate(top_smape):
        username_display = f"@{username}" if username else "Аноним"
        smape_text += (
            f"{user_scores[username]['places']['smape']}. <b>{username_display}</b> — <b>{best_smape:.2f}%</b> · попыток: {attempts}\n"
        )
        users_was_smape.append(username)

    if user_smape_pos:
        if user_name not in users_was_smape:
            smape_text += "\n—\n\n"
            smape_text += f"<b>{user_smape_pos}.</b> @{user_name} — <b>{user_smape:.2f}%</b> · попыток: {user_attempts}"
    
    await update.message.reply_text(smape_text, parse_mode='HTML')
    
    # 2. DA
    da_text = "<b>Лидерборд — Direction Accuracy</b>\n\n"
    da_text += "Доля правильно предсказанных направлений.\n"
    da_text += "Больше — лучше.\n\n"

    users_was_da = []
    
    for i, (username, best_da, smape, mae, rmse, attempts, last_attempt) in enumerate(top_da):
        username_display = f"@{username}" if username else "Аноним"
        da_text += (
            f"{user_scores[username]['places']['da']}. <b>{username_display}</b> — <b>{best_da:.1%}</b> · попыток: {attempts}\n"
        )
        users_was_da.append(username)

    if user_da_pos:
        if user_name not in users_was_da:
            da_text += "\n—\n\n"
            da_text += f"<b>{user_da_pos}.</b> @{user_name} — <b>{user_da:.1%}</b> · попыток: {user_attempts}"
    
    await update.message.reply_text(da_text, parse_mode='HTML')
    
    # 3. MAE
    mae_text = "<b>Лидерборд — MAE</b>\n\n"
    mae_text += "Средняя абсолютная ошибка.\n"
    mae_text += "Меньше — лучше.\n\n"

    users_was_mae = []
    
    for i, (username, best_mae, smape, da, rmse, attempts, last_attempt) in enumerate(top_mae):
        username_display = f"@{username}" if username else "Аноним"
        mae_text += (
            f"{user_scores[username]['places']['mae']}. <b>{username_display}</b> — <b>{best_mae:.2f}</b> · попыток: {attempts}\n"
        )
        users_was_mae.append(username)
    
    if user_mae_pos:
        if user_name not in users_was_mae:
            mae_text += "\n—\n\n"
            mae_text += f"<b>{user_mae_pos}.</b> @{user_name} — <b>{user_mae:.2f}</b> · попыток: {user_attempts}"
    
    await update.message.reply_text(mae_text, parse_mode='HTML')
    
    # 4. RMSE
    rmse_text = "<b>Лидерборд — RMSE</b>\n\n"
    rmse_text += "Ошибка с усиленным штрафом за крупные отклонения.\n"
    rmse_text += "Меньше — лучше.\n\n"

    users_was_rmse = []
    
    for i, (username, best_rmse, smape, da, mae, attempts, last_attempt) in enumerate(top_rmse):
        username_display = f"@{username}" if username else "Аноним"
        rmse_text += (
            f"{user_scores[username]['places']['rmse']}. <b>{username_display}</b> — <b>{best_rmse:.2f}</b> · попыток: {attempts}\n"
        )
        users_was_rmse.append(username)

    if user_rmse_pos:
        if user_name not in users_was_rmse:
            rmse_text += "\n—\n\n"
            rmse_text += f"<b>{user_rmse_pos}.</b> @{user_name} — <b>{user_rmse:.2f}</b> · попыток: {user_attempts}"
    
    await update.message.reply_text(rmse_text, parse_mode='HTML')
    
    # 5. TOTAL
    overall_text = "<b>Общий рейтинг</b>\n\n"
    overall_text += "Итоговая позиция по всем метрикам.\n\n"

    users_was_olverall = []
    
    for i, user in enumerate(overall_leaderboard[:5], 1):
        username_display = f"@{user['username']}" if user['username'] else "Аноним"
        
        overall_text += f"{i}. <b>{username_display}</b> — <b>{user['scores']}</b>\n"
        overall_text += f"   SMAPE: {user['places']['smape']} · Direction Accuracy: {user['places']['da']} · MAE: {user['places']['mae']} · RMSE: {user['places']['rmse']}"
        
        overall_text += "\n\n"

        users_was_olverall.append(user['username'])
    
    if user_overall_pos:
        if user_name not in users_was_olverall:
            not_top_5_user_score = user_mae_pos + user_da_pos + user_rmse_pos + user_smape_pos

            overall_text += '\n—\n\n'
            overall_text += f"{user_overall_pos}. <b>{user_name}</b> — <b>{not_top_5_user_score}</b>\n"
            overall_text += f"   SMAPE: {user_smape_pos} · Direction Accuracy: {user_da_pos} · MAE: {user_mae_pos} · RMSE: {user_rmse_pos}"

            overall_text += "\n\n"
    
    overall_text += (
        """Баллы начисляются за места в каждом лидерборде:  
1 — 1, 2 — 2, 3 — 3, 4 — 4, 5 — 5, ...  
Меньше — лучше."""
    )
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        overall_text,
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect('user_prompts.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM prompts')
    total_messages = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM prompts')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(smape) FROM prompts WHERE smape IS NOT NULL')
    avg_mape = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT MIN(smape) FROM prompts WHERE smape IS NOT NULL')
    best_mape = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT MAX(smape) FROM prompts WHERE smape IS NOT NULL')
    worst_mape = cursor.fetchone()[0] or 0
    
    conn.close()
    
    stats_text = (
        "📊 <b>СТАТИСТИКА СИСТЕМЫ</b>\n\n"
        f"<b>Пользователей:</b> {total_users}\n"
        f"<b>Всего промптов:</b> {total_messages}\n"
        f"<b>Средний SMAPE:</b> {avg_mape:.2f}%\n"
        f"<b>Лучший SMAPE:</b> {best_mape:.2f}%\n"
        f"<b>Худший SMAPE:</b> {worst_mape:.2f}%\n\n"
        f"<b>Лимит попыток:</b> {MAX_ATTEMPTS} на пользователя"
    )
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        stats_text,
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
<b>Справка</b>

<b>О боте</b>
Бот тестирует промпты для прогнозирования цен на исторических данных.  
Ты отправляешь промпт — получаешь оценку точности прогноза.

<b>Как начать</b>
1. Команда <b>/start</b> — приветствие и количество попыток  
2. Отправь текстовый промпт  
3. Получи метрики качества и позицию в рейтинге

<b>Метрики</b>

<b>SMAPE</b>  
Средняя процентная ошибка. Меньше — лучше.

<b>Direction Accuracy</b>  
Доля правильно предсказанных направлений (рост / падение). Больше — лучше.

<b>MAE</b>  
Средняя абсолютная ошибка в исходных единицах.

<b>RMSE</b>  
Метрика с усиленным штрафом за крупные ошибки.

<b>Лидерборды</b>
Команда <b>/top</b> показывает:
• рейтинг по каждой метрике  
• общий рейтинг на основе суммы мест

<b>Команды</b>
• <b>/start</b> — начать работу  
• <b>/stats</b> — общая статистика  
• <b>/top</b> — лидерборды  
• <b>/base_prompt</b> — пример промпта  
• <b>/help</b> — эта справка

<b>Ограничения</b>
• Доступно попыток: <b>{MAX_ATTEMPTS}</b>  
• Промпт должен возвращать только числа  
• Лидерборды обновляются автоматически

<b>Рекомендации</b>
• Чётко описывай логику анализа  
• Используй пример как ориентир  
• Тестируй разные подходы
""".format(MAX_ATTEMPTS=MAX_ATTEMPTS)

    keyboard = [
        [KeyboardButton("✨ Пример промпта"), KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт"), KeyboardButton("❓ Помощь")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        help_text,
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = await get_used_attempts(user.id)
    
    if used_attempts >= MAX_ATTEMPTS:
        await update.message.reply_text(
            "❌ <b>Попытки закончились</b>\n\n"
            "Вы можете посмотреть топ результатов и статистику!",
            parse_mode='HTML'
        )
        return
    
    keyboard = [
        [KeyboardButton("❓ Помощь"), KeyboardButton("✨ Пример промпта")],
        [KeyboardButton("📝 Как это работает?")],
        [KeyboardButton("🏆 Топ промптов"), KeyboardButton("📊 Статистика")],
        [KeyboardButton("🚀 Тестировать промпт")],
    ]

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "🤖 <b>Я понимаю только текстовые промпты!</b>\n\n"
        "Отправьте текстовую инструкцию для нейросети. "
        "Например: 'Проанализируй тренды и предскажи будущие цены'\n\n"
        "Или нажмите кнопку '📝 Как это работает?' для подробного руководства.",
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def run_bot():
    await db_controller.connect()
    await init_db()

    worker = asyncio.create_task(create_worker())
    
    application = Application.builder().token(BOT_TOKEN).connect_timeout(30).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("top", show_top_prompts))
    application.add_handler(CommandHandler("base_prompt", show_example_prompt))
    
    application.add_handler(MessageHandler(
        filters.Text(["✅ Да, тестировать", "✏️ Переписать"]), 
        handle_confirmation
    ))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_other_messages))
    
    print("Бот запущен...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    await asyncio.Event().wait()

async def main_async():
    while True:
        try:
            await run_bot()

        except (NetworkError, ConnectionError) as e:
            # print(f"Потеря сети: {e}")
            print("Жду интернет и пробую переподключиться через 5 сек...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"Ошибка бота: {e}")
            print("Перезапуск через 5 сек...")
            await asyncio.sleep(5)

def main():
    asyncio.run(main_async())
    

if __name__ == '__main__':
    main()