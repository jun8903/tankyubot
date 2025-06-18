from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import pandas as pd

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SERVICE_ACCOUNT_FILE = '/Users/jungouchi/Desktop/tankyubot/creds.json'
SPREADSHEET_ID = '1MiIs_URQ4Td_h9YRwN1M8SYu_NY2ZiansEfizmb8JRY'
RANGE_NAME = 'シート1!A:C'

def load_data_from_gspread():
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
        return pd.DataFrame()

    df = pd.DataFrame(values[1:], columns=values[0])
    return df

def create_indexes(df):
    # ここにインデックス作成の処理を書く
    pass
