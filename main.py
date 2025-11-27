import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import sqlite3
import os

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = "8344196862:AAGFRBrgddxNbPMhxXYpZs3lIgVHWauxxdY"
MAX_ATTEMPTS = 5  # Количество попыток

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('user_messages.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            message_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_attempts (
            user_id INTEGER PRIMARY KEY,
            attempts_used INTEGER DEFAULT 0,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Сохранение сообщения в БД
def save_message(user_id, username, message_text):
    conn = sqlite3.connect('user_messages.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO messages (user_id, username, message_text) VALUES (?, ?, ?)',
        (user_id, username, message_text)
    )
    
    # Обновляем счетчик попыток
    cursor.execute('''
        INSERT OR REPLACE INTO user_attempts (user_id, attempts_used, last_activity)
        VALUES (?, COALESCE((SELECT attempts_used FROM user_attempts WHERE user_id = ?) + 1, 1), CURRENT_TIMESTAMP)
    ''', (user_id, user_id))
    
    conn.commit()
    conn.close()

# Получение количества использованных попыток
def get_used_attempts(user_id):
    conn = sqlite3.connect('user_messages.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT attempts_used FROM user_attempts WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else 0

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = get_used_attempts(user.id)
    remaining_attempts = MAX_ATTEMPTS - used_attempts
    
    welcome_text = (
        f"Привет, {user.first_name}! 👋\n\n"
        f"У тебя есть {MAX_ATTEMPTS} попыток, чтобы отправить сообщения.\n"
        f"Осталось попыток: {remaining_attempts}\n\n"
        "Просто напиши любое сообщение, и я его сохраню!"
    )
    
    await update.message.reply_text(welcome_text)

# Обработка текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text
    
    used_attempts = get_used_attempts(user.id)
    
    if used_attempts >= MAX_ATTEMPTS:
        await update.message.reply_text(
            f"❌ Извини, но твои {MAX_ATTEMPTS} попыток закончились!\n"
            "Больше сообщения не принимаются."
        )
        return
    
    # Сохраняем сообщение
    save_message(user.id, user.username, message_text)
    
    remaining_attempts = MAX_ATTEMPTS - (used_attempts + 1)
    
    if remaining_attempts > 0:
        response_text = (
            f"✅ Сообщение сохранено!\n"
            f"Осталось попыток: {remaining_attempts}"
        )
    else:
        response_text = (
            f"✅ Сообщение сохранено!\n"
            f"❌ Это была твоя последняя попытка. Больше сообщения не принимаются."
        )
    
    await update.message.reply_text(response_text)

# Обработка других типов сообщений
async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    used_attempts = get_used_attempts(user.id)
    
    if used_attempts >= MAX_ATTEMPTS:
        await update.message.reply_text(
            f"❌ Извини, но твои {MAX_ATTEMPTS} попыток закончились!"
        )
        return
    
    await update.message.reply_text(
        "❌ Я принимаю только текстовые сообщения!\n"
        f"Эта попытка не засчитана. Осталось попыток: {MAX_ATTEMPTS - used_attempts}"
    )

# Функция для просмотра статистики (только для администратора)
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Здесь можно добавить проверку на администратора
    conn = sqlite3.connect('user_messages.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM messages')
    total_messages = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM messages')
    total_users = cursor.fetchone()[0]
    
    conn.close()
    
    stats_text = (
        f"📊 Статистика бота:\n"
        f"• Всего сообщений: {total_messages}\n"
        f"• Всего пользователей: {total_users}\n"
        f"• Максимум попыток: {MAX_ATTEMPTS}"
    )
    
    await update.message.reply_text(stats_text)

def main():
    # Инициализация базы данных
    init_db()
    
    # Создание приложения
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    
    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_other_messages))
    
    # Запуск бота
    print("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()