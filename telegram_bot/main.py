import telebot
import requests
import hashlib
from recongize import image_train, image_test
import glob
import os

TOKEN = "547390748:AAFUYkFTIu2otwjv7MWLb1zuODMhJ8eDAEI"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(content_types = ['photo'])
def hand_photo(message):
    original_size = message.photo[len(message.photo) - 1]
    file_info = bot.get_file(original_size.file_id)
    file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(TOKEN, file_info.file_path))
    h256 = hashlib.sha256(file.content)
    image_name = h256.hexdigest() + ".jpg"
    with open(image_name, "wb") as f:
        f.write(file.content)
        print("file:", image_name, "is sucsessfuly created")
    if glob.glob("bof.pkl") == []:
        bot.reply_to(message, "Что это за картинка???")
    else:
        answer = image_test(image_name)
        bot.reply_to(message, "похоже, это " + answer + ".Если нет, то пришли описание, что это")

@bot.message_handler(content_types = ['text'])
def hand_text(message):
    for filename in glob.glob('*.jpg'):
        image_train(filename, message.text)
        os.remove(filename)
        print("file:", filename, "is sucsessfuly deleted")



bot.polling()
