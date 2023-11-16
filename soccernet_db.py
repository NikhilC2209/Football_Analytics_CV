import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="SoccerNet_DB")

from dotenv import load_dotenv
import os

load_dotenv()

mySoccerNetDownloader.password = os.getenv("SOCCERNET_DB_PASSWD")

mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])