import os
import openpyxl
import pandas as pd
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import torch


def ensure_header(sheet, columns):
    first_row = sheet.row_values(1)
    if not first_row or all(cell == "" for cell in first_row):
        sheet.update("A1", [columns], value_input_option="RAW")


def to_python_safe(x):
    if x is None:
        return ""

    # torch.device, torch.dtype
    if isinstance(x, (torch.device, torch.dtype)):
        return str(x)

    # torch.Tensor
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)})"

    # numpy scalar
    if hasattr(x, "item") and not isinstance(x, (list, tuple, np.ndarray)):
        return x.item()

    # list / array / tuple
    if isinstance(x, (list, tuple, np.ndarray)):
        return str(x)

    # NaN (scalar only)
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    return x


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gspread_client():
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )

            auth_url, _ = flow.authorization_url(
                prompt="consent", access_type="offline"
            )

            print("\n Open this URL in your browser:\n")
            print(auth_url)

            code = os.environ.get("GOOGLE_OAUTH_CODE")
            if not code:
                raise RuntimeError(
                    "GOOGLE_OAUTH_CODE not set. "
                    "Please authorize in browser and export the code."
                )

            flow.fetch_token(code=code)
            creds = flow.credentials

        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)

    return gspread.authorize(creds)


# automate create sheet name
def get_or_create_worksheet(client, spreadsheet_id, sheet_name, rows=1000, cols=50):
    sh = client.open_by_key(spreadsheet_id)

    try:
        sheet = sh.worksheet(sheet_name)
    except Exception:
        sheet = sh.add_worksheet(title=sheet_name, rows=rows, cols=cols)

    return sheet


def save_experiment_to_gsheet_oauth(
    args,
    metrics,
    spreadsheet_id="1k0e4gQpWVylg6NNedfhYp71rTvPoLvvxsFADsa2QW4U",
    sheet_name="Total",
):
    client = get_gspread_client()
    sheet = get_or_create_worksheet(client, spreadsheet_id, sheet_name)

    row = {**vars(args), **metrics}
    df = pd.DataFrame([row])
    columns = df.columns.tolist()
    ensure_header(sheet, columns)

    existing_rows = sheet.get_all_values()

    if not existing_rows:
        sheet.append_row(df.columns.tolist())
    row = [to_python_safe(x) for x in df.iloc[0]]

    sheet.append_row(row, value_input_option="USER_ENTERED")


def save_experiment_to_excel(args, metrics, excel_path="experiments.xlsx"):
    # args: Namespace (self.args)
    # metrics: dict {mae, mse, rmse, mape, mspe}

    row = {**vars(args), **metrics}
    df_new = pd.DataFrame([row])

    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
