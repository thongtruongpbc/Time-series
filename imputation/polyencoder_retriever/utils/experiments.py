import os
import pandas as pd
import numpy as np
import torch
import pygsheets

# --- Data Transformation Layer (Single Responsibility) ---


def to_python_safe(x):
    """
    Serializes complex data types (Tensors, Numpy, etc.)
    into a format compatible with Google Sheets/Excel.
    """
    if x is None:
        return ""

    # Handle PyTorch specific types
    if isinstance(x, (torch.device, torch.dtype)):
        return str(x)
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)})"

    # Handle Numpy scalars and arrays
    if hasattr(x, "item") and not isinstance(x, (list, tuple, np.ndarray)):
        return x.item()
    if isinstance(x, (list, tuple, np.ndarray)):
        return str(x)

    # Handle NaN values
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    return x


# --- Authentication Layer (Open/Closed Principle) ---


def get_gspread_client():
    """
    Authorizes using OAuth2.
    It will open a browser for the first run and cache the token
    locally to avoid future manual sign-ins.
    """
    # pygsheets looks for 'client_secret.json' and stores credentials in a local file
    return pygsheets.authorize(client_secret="client_secret.json")


# --- Storage Layer (Interface Segregation) ---


def get_or_create_worksheet(client, spreadsheet_id, sheet_name):
    """
    Opens the spreadsheet and ensures the worksheet exists.
    """
    spreadsheet = client.open_by_key(spreadsheet_id)
    try:
        return spreadsheet.worksheet_by_title(sheet_name)
    except pygsheets.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(sheet_name)


def save_experiment_to_gsheet_oauth(
    args,
    metrics,
    spreadsheet_id="1k0e4gQpWVylg6NNedfhYp71rTvPoLvvxsFADsa2QW4U",
    sheet_name="Total",
):
    """
    Saves experiment results to Google Sheets using pygsheets.
    """
    client = get_gspread_client()
    worksheet = get_or_create_worksheet(client, spreadsheet_id, sheet_name)

    # Prepare data row
    combined_data = {**vars(args), **metrics}
    # Ensure all values are serialized for Google Sheets
    serialized_row = [to_python_safe(v) for v in combined_data.values()]
    headers = list(combined_data.keys())

    # Check if header exists by looking at cell A1
    existing_headers = worksheet.get_row(1, include_tailing_empty=False)
    if not existing_headers:
        worksheet.set_dataframe(pd.DataFrame(columns=headers), start="A1")

    # Append the new row to the end of the table
    # pygsheets append_table automatically finds the last row
    worksheet.append_table(values=[serialized_row], start="A1", overwrite=False)
    print(f"Experiment data successfully saved to Google Sheet: {sheet_name}")


def save_experiment_to_excel(args, metrics, excel_path="experiments.xlsx"):
    """
    Saves experiment results to a local Excel file.
    """
    row_dict = {**vars(args), **metrics}
    # Clean data before creating DataFrame
    cleaned_row = {k: to_python_safe(v) for k, v in row_dict.items()}
    df_new = pd.DataFrame([cleaned_row])

    if os.path.exists(excel_path):
        try:
            df_old = pd.read_excel(excel_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(excel_path, index=False)
