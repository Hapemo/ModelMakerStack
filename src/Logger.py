from datetime import datetime
import os

class Colors:
  RESET = "\033[0m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"
  RED = "\033[31m"
  GREEN = "\033[32m"
  YELLOW = "\033[33m"
  BLUE = "\033[34m"
  MAGENTA = "\033[35m"
  CYAN = "\033[36m"
  WHITE = "\033[37m"

LOGGER_INITIALIZED = False

class Logger:
  filepath = "log"
  printConsole = False

  @staticmethod
  def Initialize():
    global LOGGER_INITIALIZED
    if LOGGER_INITIALIZED: return
    LOGGER_INITIALIZED = True
    if os.path.exists(Logger.filepath):
      for filename in os.listdir(Logger.filepath):
        file_path = os.path.join(Logger.filepath, filename)
        if os.path.isfile(file_path):
          os.unlink(file_path)
  
  @staticmethod
  def SetFilePath(path:str):
    Logger.filepath = path

  @staticmethod
  def LogCustom(category:str, message:str, filename:str = "", timestamp:bool = True, logInAll:bool = True):
    category = category.upper()

    if not os.path.exists(Logger.filepath): os.makedirs(Logger.filepath)

    if len(filename) == 0:
      filename = category + ".log"

    color = Colors.WHITE
    if category == "WARNING": color = Colors.YELLOW
    elif category == "ERROR": color = Colors.RED
      
    log_message = f"[{Logger.GetTime()}] {color}[{category}]{Colors.RESET}: {message}\n"
    
    # Write to the specific log file
    with open(os.path.join(Logger.filepath, filename), 'a', encoding='utf-8') as out_file:
      out_file.write(log_message)
    
    # Write to the ALL.log file if needed
    if logInAll:
      with open(os.path.join(Logger.filepath, "ALL.log"), 'a', encoding='utf-8') as all_log:
        all_log.write(log_message)
    
    # Print to console if needed
    if Logger.printConsole:
      print(log_message, end='')

    return
  
  @staticmethod
  def LogInfo(message:str, filename:str = "", timestamp:bool = True, logInAll:bool = True):
    Logger.LogCustom("INFO", message, filename, timestamp, logInAll)
  
  @staticmethod
  def LogWarning(message:str, filename:str = "", timestamp:bool = True, logInAll:bool = True):
    Logger.LogCustom("WARNING", message, filename, timestamp, logInAll)

  @staticmethod
  def LogError(message:str, filename:str = "", timestamp:bool = True, logInAll:bool = True):
    Logger.LogCustom("ERROR", message, filename, timestamp, logInAll)

  @staticmethod
  def GetTime():
    # Get the current date and time
    now = datetime.now()

    # Format the datetime object as a string
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_now

# Ensure the Initialize method is called when the Logger class is used for the first time









