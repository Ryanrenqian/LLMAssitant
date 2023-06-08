import hupper
import webui
from dotenv import load_dotenv,find_dotenv
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    reloader = hupper.start_reloader('webui.main')
    webui.main()