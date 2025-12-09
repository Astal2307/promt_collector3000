import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import ReplyKeyboardMarkup, KeyboardButton
import sqlite3
import os
from test_test_prompt import Tester
import requests
from db_controller import DBController
import asyncio
import aiosqlite
from datetime import datetime

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

OAUTH_TOKEN = "your_oauth_token"
IAM_TOKEN = "" #get_iam_token(OAUTH_TOKEN)
FOLDER_ID = "your_folder_id"
MODEL_URI = "your_model_uri"
BOT_TOKEN = "your_bot_token"
MAX_ATTEMPTS = 100

db_controller = DBController()
# tester = Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller)

async def init_db():
    async with aiosqlite.connect('user_prompts.db') as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                username TEXT,
                prompt TEXT,
                mape FLOAT,
                direction_accuracy FLOAT
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


async def save_prompt(user_id, username, prompt, overall_metrics):
    async with aiosqlite.connect('user_prompts.db') as conn:
        await conn.execute(
            'INSERT INTO prompts (user_id, username, prompt, mape, direction_accuracy) VALUES (?, ?, ?, ?, ?)',
            (user_id, username, prompt, round(overall_metrics.avg_mape, 3), round(overall_metrics.avg_direction_accuracy, 3))
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


async def get_top_users_by_mape(limit=5):
    async with aiosqlite.connect('user_prompts.db') as conn:
        cursor = await conn.execute('''
            SELECT 
                p.username,
                p.mape as best_mape,
                p.direction_accuracy,
                g.attempts,
                g.last_attempt
            FROM prompts p
            JOIN (
                SELECT 
                    user_id,
                    MIN(mape) as min_mape,
                    COUNT(*) as attempts,
                    MAX(timestamp) as last_attempt
                FROM prompts 
                WHERE username IS NOT NULL AND mape IS NOT NULL
                GROUP BY user_id
            ) g ON p.user_id = g.user_id AND p.mape = g.min_mape
            ORDER BY p.mape ASC
            LIMIT ?
        ''', (limit,))
        
        results = await cursor.fetchall()
        return results


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = await get_used_attempts(user.id)
    remaining_attempts = MAX_ATTEMPTS - used_attempts
    
    keyboard = [
        [KeyboardButton("🏆 Топ промптов")],
        [KeyboardButton("📊 Статистика")],
        [KeyboardButton("❓ Помощь")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    welcome_text = (
        f"{user.first_name}, тестируйте промпты для прогнозирования цен.\n\n"
        
        "Итоговый промпт состоит из вашего запроса и добавочной части:\n\n"

        "-----------------------------------------------------------------\n"
        "КОНТЕКСТ ДАННЫХ:\n"
        "timestamp, open, high, low, close, volume\n\n"

        "ИНСТРУКЦИИ:\n"
        "   - Проанализируй исторические данные\n"
        "   - Предскажи цену закрытия следующей свечи\n"
        "   - Верни только числовое значение в формате: 123.45\n"
        "   - Не добавляй пояснений, текста или символов\n"
        "-----------------------------------------------------------------\n\n"

        "Примеры промптов:\n"
        "• Учти тренд и волатильность\n" 
        "• Проанализируй ценовые паттерны\n"
        "• Сделай прогноз с учетом объемов торгов\n\n"
        
        f"Осталось попыток: {remaining_attempts}/{MAX_ATTEMPTS}\n\n"
        
        "Отправьте промпт для тестирования."
    )
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text
    
    if message_text == "🏆 Топ промптов":
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
            f"❌ Извини, но твои {MAX_ATTEMPTS} попыток закончились!\n"
            "Больше сообщения не принимаются."
        )
        return
    
    sent_message = await update.message.reply_text("🔍 Тестирую промпт... Это может занять несколько минут.")
    
    asyncio.create_task(
        process_prompt_testing(update, context, user, message_text, used_attempts, sent_message)
    )

async def process_prompt_testing(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                user, message_text: str, used_attempts: int, sent_message):
    try:
        test_dataset = await db_controller.sample_data(num_samples=1)
        async with Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller) as tester:
            results = await tester.test_prompt_on_dataset(
                user_prompt=message_text,
                test_dataset=test_dataset,
                horizon=5
            )

            metrics = results['metrics']
            mape = metrics.avg_mape
            direction_accuracy = metrics.avg_direction_accuracy

            await save_prompt(user.id, user.username, message_text, metrics)
            
            remaining_attempts = MAX_ATTEMPTS - (used_attempts + 1)
            
            if remaining_attempts > 0:
                response_text = (
                    f"✅ Промпт протестирован!\n\n"
                    f"📊 Результаты:\n"
                    f"• MAPE: {mape:.2f}%\n"
                    f"• Accuracy направления: {direction_accuracy:.1%}\n"
                    f"• Осталось попыток: {remaining_attempts}"
                )
            else:
                response_text = (
                    f"✅ Промпт протестирован!\n\n"
                    f"📊 Результаты:\n"
                    f"• MAPE: {mape:.2f}%\n"
                    f"• Accuracy направления: {direction_accuracy:.1%}\n"
                    f"❌ Это была твоя последняя попытка."
                )
            
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=sent_message.message_id,
                text=response_text
            )
            
    except Exception as e:
        print(f"Ошибка тестирования промпта: {e}")
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=sent_message.message_id,
            text=response_text
        )


async def show_top_prompts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    top_users = await get_top_users_by_mape(limit=5)
    
    if not top_users:
        await update.message.reply_text("📭 Пока нет данных о тестировании промптов.")
        return
    
    top_text = "🏆 ТОП-5 ЛУЧШИХ ПОЛЬЗОВАТЕЛЕЙ ПО MAPE\n\n"
    top_text += "🥇 ЛУЧШИЕ РЕЗУЛЬТАТЫ:\n\n"
    
    for i, (username, best_mape, direction_acc, attempts, last_attempt) in enumerate(top_users):
        top_text += (
            f"{i + 1}. @{username if username else 'Аноним'}\n"
            f"   MAPE: {best_mape:.2f}%\n"
            f"   direction_accuracy: {direction_acc:.1%}\n"
            f"   Попыток: {attempts}\n\n"
        )
    
    await update.message.reply_text(top_text)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect('user_prompts.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM prompts')
    total_messages = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM prompts')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(mape) FROM prompts WHERE mape IS NOT NULL')
    avg_mape = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT MIN(mape) FROM prompts WHERE mape IS NOT NULL')
    best_mape = cursor.fetchone()[0] or 0
    
    conn.close()
    
    stats_text = (
        f"📊 СТАТИСТИКА БОТА:\n\n"
        f"• Всего промптов: {total_messages}\n"
        f"• Уникальных пользователей: {total_users}\n"
        f"• Средний MAPE: {avg_mape:.2f}%\n"
        f"• Лучший MAPE: {best_mape:.2f}%\n"
        f"• Лимит попыток: {MAX_ATTEMPTS}"
    )
    
    await update.message.reply_text(stats_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "❓ ПОМОЩЬ ПО БОТУ\n\n"
        "🤖 Этот бот тестирует промпты для прогнозирования цен:\n\n"
        "📝 Отправь промпт - я протестирую его на исторических данных\n"
        "📊 Увидишь метрики MAPE и точность направления\n"
        "🏆 Сравни свои результаты с другими в топе\n\n"
        "📋 КОМАНДЫ:\n"
        "• Произвольный промпт - тестирование\n"
        "• 🏆 Топ промптов - лучшие результаты\n"
        "• 📊 Статистика - общая статистика\n"
        "• ❓ Помощь - это сообщение\n\n"
        "💡 MAPE - средняя процентная ошибка прогноза (чем меньше - тем лучше)"
    )
    
    await update.message.reply_text(help_text)

async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = await get_used_attempts(user.id)
    
    if used_attempts >= MAX_ATTEMPTS:
        await update.message.reply_text(
            f"❌ Извини, но твои {MAX_ATTEMPTS} попыток закончились!"
        )
        return
    
    await update.message.reply_text(
        "❌ Я принимаю только текстовые сообщения!\n"
        f"Эта попытка не засчитана. Осталось попыток: {MAX_ATTEMPTS - used_attempts}"
    )

async def main_async():
    await db_controller.connect()
    await init_db()
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("top", show_top_prompts))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_other_messages))
    
    print("Бот запущен...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    await asyncio.Event().wait()

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()

    """async def proceed():
        test_dataset = await db_controller.sample_data(num_samples=10)
        async with Tester(IAM_TOKEN, FOLDER_ID, MODEL_URI, db_controller) as tester:
            results = await tester.test_prompt_on_dataset(
                user_prompt="123",
                test_dataset=test_dataset,
                horizon=5
            )
            metrics = results['metrics']
            mape = metrics.avg_mape
            direction_accuracy = metrics.avg_direction_accuracy

        print(mape, direction_accuracy)
    

    asyncio.run(proceed())"""
    