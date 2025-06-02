# Copyright (C) Kevin Jin, <kevinjin21@gmail.com>, 2025

from pypdf import PdfReader
import tabula, os, shutil, sqlite3, re, glob, warnings
import numpy as np
import pandas as pd
from utils.record_events import RecordLogs
import pymupdf
from datetime import datetime
from typing import List, Dict
from utils.category_llm import categorize_transaction

__version__ = '1.1'
# v1.1: add llm-generated category column to db

LOG_NAME = 'invoice_parser'
LOG_LOC = f"logs/{LOG_NAME}-log-{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = RecordLogs(__version__, LOG_LOC, os.path.basename(__file__)).logger
current_year = datetime.now().year

def create_tables():
    conn = sqlite3.connect('finance_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bank_summary (
            invoice_id TEXT PRIMARY KEY,
            beginning_balance REAL,
            ending_balance REAL,
            deposits REAL,
            withdrawals REAL,
            date_start DATE,
            date_end DATE,
            account_number TEXT,
            created_on DATETIME DEFAULT (datetime('now', 'localtime')),
            modified_on DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS card_summary (
            invoice_id TEXT PRIMARY KEY,
            card_number TEXT,
            previous_balance REAL,
            current_balance REAL,
            date_start DATE,
            date_end DATE,
            cash_advances REAL,
            balance_transfers REAL,
            fees REAL,
            interest REAL,
            available_credit REAL,
            credit_limit REAL,
            created_on DATETIME DEFAULT (datetime('now', 'localtime')),
            modified_on DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id TEXT,
            date TEXT,
            desc TEXT,
            transaction_amt REAL,
            res_balance REAL,
            adjusted_date DATE,
            created_on DATETIME DEFAULT (datetime('now', 'localtime')),
            modified_on DATETIME DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (invoice_id) REFERENCES summary(invoice_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_inv_type(reader):
    page = reader.pages[0]
    content = page.extract_text()

    '''_tmp_df = tabula.read_pdf(file_path, pages=1, stream=True, relative_area=True, area=(0, 18, 100, 100),
                              pandas_options={'header': None}, silent=True)
    _tmp_df[0].dropna(how='all', inplace=True)
    _tmp_df[0] = _tmp_df[0].fillna('')
    _tmp_df[0]['res'] = _tmp_df[0].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    for t in _tmp_df[0].res.to_list():
        find_checking = re.findall('CHECKING SUMMARY', content)
        find_credit = re.findall('ACCOUNT_SUMMARY', content)
        if len(find_checking) > 0:
            return 'bank'
        if len(find_credit) > 0:
            return 'card'
    '''
        
    find_checking = re.findall('Chase Total Checking', content)
    if len(find_checking) > 0:
        return 'bank'
    find_credit = re.findall('ACCOUNT SUMMARY', content)
    if len(find_credit) > 0:
        return 'card'

    return 'unknown'

def get_page_content(page_no, total_pages, reader, invoice_type):
    if invoice_type == 'bank':
        pattern = 'A Monthly Service Fee'
    elif invoice_type == 'card':
        pattern = 'INTEREST CHARGES'
    else:
        return ''

    page = reader.pages[page_no - 1]
    content = page.extract_text()      

    # skip pages until reaching card activity section
    if invoice_type == 'card':
        start = re.findall('ACCOUNT ACTIVITY', content)
        while len(start) == 0:
            if page_no == total_pages:
                logger.info(f"Couldn't find transaction details.")
                return ''
            page_no += 1
            page = reader.pages[page_no - 1]
            content = page.extract_text()
            start = re.findall('ACCOUNT ACTIVITY', content)

    result = content 
    check = len(re.findall(pattern, content))

    # end page_content when reaching the search word
    while check == 0 and page_no < total_pages:
        page_no += 1
        page = reader.pages[page_no - 1]
        content = page.extract_text()
        result += content
        check = len(re.findall(pattern, content))

    return result

def get_bank_summary_details(reader):
    page = reader.pages[0]
    content = page.extract_text()

    fields = {
        'beginning': {
            'regex': 'Beginning Balance\s+\$([\d.,]+)',
            'expected_results': ['balance']
        },
        'ending': {
            'regex': 'Ending Balance\s+\$([\d.,]+)',
            'expected_results': ['balance']
        },
        'deposits': {
            'regex': 'Deposits and Additions\s+([\d,.]+)',
            'expected_results': ['additions']
        },
        'electronic': {
            'regex': 'Electronic Withdrawals\s+\-([\d,.]+)',
            'expected_results': ['withdrawals']
        },
        'date': {
            #'regex': '(?P<start>.*)\s+through\s+(?P<end>.*)',
            'regex': '\\n(?P<start>[\w\s,]+)\s*through\s*(?P<end>[\w\s,]+)\\n',
            'expected_results': ['start', 'end']
        },
        'account': {
            'regex': '(\d+)\\nCUSTOMER',
            'expected_results': ['number']
        }
    }

    _rs_dct = {}
    try:
        for k, v in fields.items():
            find = re.findall(v['regex'], content)
            if len(find) > 0:
                f = list(find[0]) if type(find[0]) is tuple else find
                dct = dict(
                    zip(list(map(lambda x: f'{k}_{x}', v['expected_results'])), f[-len(v['expected_results']):]))
                _rs_dct.update(dct)
    except Exception as e:
        logger.info(f"Couldn't read summary: {e}")
        return pd.DataFrame()

    df = pd.DataFrame([_rs_dct])

    # Ensure required columns exist with float 0.0 default
    required_cols = {
        'beginning_balance': 0.0,
        'ending_balance': 0.0,
        'deposits_additions': 0.0,
        'electronic_withdrawals': 0.0
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val

    # float columns
    convert_cols = ['beginning_balance', 'ending_balance', 'deposits_additions', 'electronic_withdrawals']
    df.loc[:, df.columns.isin(convert_cols)] = df.loc[:, df.columns.isin(convert_cols)].replace('[\,\$\s]', '',
                                                                                                regex=True).fillna(0).astype(float)
    
    # negative amounts (withdrawals)
    if 'electronic_withdrawals' in df.columns:
        df['electronic_withdrawals'] = df['electronic_withdrawals'] * -1
    
    # clean date columns
    convert_cols = ['date_start', 'date_end']
    for col in convert_cols:
        df[col] = df[col].str.strip()
        df[col] = pd.to_datetime(df[col], format='%B %d, %Y', errors='coerce')  # Convert to datetime

    # rename columns
    df = df.rename(columns={
        'deposits_additions': 'deposits',
        'electronic_withdrawals': 'withdrawals'
    })
    
    # generate invoice id: last 4 digits of account number + start date
    df['invoice_id'] = df['account_number'].str[-4:] + '_' + df['date_start'].dt.strftime('%Y%m%d')
    return df

def get_card_summary_details(reader):
    page = reader.pages[0]
    content = page.extract_text()

    fields = {
        'card': {
            'regex': 'X{4}\s+(\d{4})',
            'expected_results': ['number']
        },
        'previous': {
            'regex': 'Previous Balance\s+\$([\d,.]+)',
            'expected_results': ['balance']
        },
        'current': {
            'regex': 'Purchases\s+\+\$([\d,.]+)',
            'expected_results': ['balance']
        },
        'date': {
            'regex': 'Opening/Closing Date\s+(?P<start>\d+/\d+/\d+)[\s\-]+(?P<end>\d+/\d+/\d+)',
            'expected_results': ['start', 'end']
        },
        'cash': {
            'regex': 'Cash Advances\s+\$(-?[\d,.]+)',
            'expected_results': ['advances']
        },
        'balance': {
            'regex': 'Balance Transfers\s+\$(-?[\d,.]+)',
            'expected_results': ['transfers']
        },
        'fees': {
            'regex': 'Fees Charged\s+\$(-?[\d,.]+)',
            'expected_results': ['charged']
        },
        'interest': {
            'regex': 'Interest Charged\s+\$(-?[\d,.]+)',
            'expected_results': ['charged']
        },
        'available': {
            'regex': 'Available Credit\s+\$([\d,.]+)',
            'expected_results': ['credit']
        },
        'credit': {
            'regex': 'Credit Access Line\s+\$([\d,.]+)',
            'expected_results': ['limit']
        },
    }

    _rs_dct = {}
    try:
        for k, v in fields.items():
            find = re.findall(v['regex'], content)
            if len(find) > 0:
                f = list(find[0]) if type(find[0]) is tuple else find
                dct = dict(
                    zip(list(map(lambda x: f'{k}_{x}', v['expected_results'])), f[-len(v['expected_results']):]))
                _rs_dct.update(dct)
    except Exception as e:
        logger.info(f"Couldn't read summary: {e}")
        return pd.DataFrame()

    df = pd.DataFrame([_rs_dct])

    # Ensure required columns exist with float 0.0 default
    required_cols = {
        'previous_balance': 0.0,
        'current_balance': 0.0,
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
    
    # rename columns to match database schema
    df = df.rename(columns={
        'fees_charged': 'fees',
        'interest_charged': 'interest'
    })

    # float columns
    convert_cols = ['previous_balance', 'current_balance', 'cash_advances', 'balance_transfers', 'fees', 'interest', 'available_credit', 'credit_limit']
    df.loc[:, df.columns.isin(convert_cols)] = df.loc[:, df.columns.isin(convert_cols)].replace('[\,\$\s]', '',
                                                                                                regex=True).fillna(0).astype(float)
    
    # clean date columns
    convert_cols = ['date_start', 'date_end']
    for col in convert_cols:
        df[col] = df[col].str.strip()
        df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')
    # generate invoice id: last 4 digits of account number + start date
    df['invoice_id'] = df['card_number'][0] + '_' + df['date_start'].dt.strftime('%Y%m%d')
    return df

def get_bank_line_items(page_content):
    fields = {
        'withdrawal': {
            #'regex': '(?P<date>\d{2}\/\d{2})\s+(?P<desc>.*?)\s+(?P<amt>[-\d,]+\.\d{2})\s+(?P<bal>[\d,.]+)',
            'regex': '(?P<date>\d{2}\/\d{2})\s+(?P<desc>.*?)\s+(?P<amt>\-?\s*[\d,]+\.\d{2})\s+(?P<bal>[\d,.]+)\\n',
            'expected_results': ['date', 'desc', 'amt', 'bal']
        },
        'deposit': {
            'regex': '(?P<date>\d{2}\/\d{2})\s+(?!.*-)(?P<desc>.*?)\s+(?P<bal>[\d,.]+)\\n',
            'expected_results': ['date', 'desc', 'bal']
        }
    }

    df = pd.DataFrame()
    try:
        _rs_dct = {}
        for k, v in fields.items():
            find = re.findall(v['regex'], page_content)
            if len(find) > 0:
                for item in find:
                    dct = dict(
                    zip(list(map(lambda x: f'{k}_{x}', v['expected_results'])), item))
                    df = pd.concat([df, pd.DataFrame([dct])], ignore_index=True) 
    except Exception as e:
        logger.info(f"Couldn't read transaction line item: {e}")
        return pd.DataFrame()
    
    if 'deposit_bal' in df.columns:
        deposits = re.findall('Beginning Balance.*?\\n([\d,.\\n]+)\\nEnding Balance', page_content)
        if len(deposits) == 0:
            logger.info("Deposit amounts not found despite being present in the invoice.")
            return pd.DataFrame()
        
        deposits = deposits[0].split('\n')
        if len(deposits) != df['deposit_bal'].notnull().sum():
            # number of bolded deposits should match number of line item deposits
            logger.info("Number of deposit amounts does not match the number of line items.")
            return pd.DataFrame()
        extra_rows = len(df) - len(deposits)
        deposits = [0]*extra_rows + deposits # add padding for withdrawal rows
        df['deposit_amt'] = deposits

    # float columns
    convert_cols = ['withdrawal_amt', 'withdrawal_bal', 'deposit_bal', 'deposit_amt']
    df.loc[:, df.columns.isin(convert_cols)] = df.loc[:, df.columns.isin(convert_cols)].replace('[\,\$\s]', '',
                                                                                            regex=True).fillna(0).astype(float)

       # combining withdrawal and deposit columns - floats
    df['res_balance'] = df.get('withdrawal_bal', pd.Series([0]*len(df))).fillna(0) + \
                       df.get('deposit_bal', pd.Series([0]*len(df))).fillna(0)
    df['transaction_amt'] = df.get('withdrawal_amt', pd.Series([0]*len(df))).fillna(0) + \
                           df.get('deposit_amt', pd.Series([0]*len(df))).fillna(0)
    df.drop(columns=['withdrawal_bal', 'deposit_bal', 'withdrawal_amt', 'deposit_amt'], inplace=True, errors='ignore')

    # combining columns - strings
    df['desc'] = df.get('withdrawal_desc', pd.Series([''] * len(df))).fillna('') + \
                 df.get('deposit_desc', pd.Series([''] * len(df))).fillna('')
    df['date'] = df.get('withdrawal_date', pd.Series([''] * len(df))).fillna('') + \
                 df.get('deposit_date', pd.Series([''] * len(df))).fillna('')
    df.drop(columns=['withdrawal_desc', 'withdrawal_date', 'deposit_desc', 'deposit_date'], inplace=True, errors='ignore')

    '''# clean up column names
    _l = list(map(lambda x: x.replace("line_item_", ''), df.columns.to_list()))
    df.rename(columns=dict(zip(df.columns, _l)), inplace=True)'''  

    # sort by date -> format as datetime first
    df['adjusted_date'] = df['date'].apply(
    lambda x: f"{current_year - 1}/{x}" if x.startswith('12/') else f"{current_year}/{x}"
    )
    df['adjusted_date'] = pd.to_datetime(df['adjusted_date'], format='%Y/%m/%d')
    df = df.sort_values(by='adjusted_date').reset_index(drop=True)

    # Drop the adjusted_date column if no longer needed
    # df.drop(columns=['adjusted_date'], inplace=True)

    return df

def get_card_line_items(page_content):
    fields = {
        'transaction': {
            'regex': '(?P<date>\d{2}\/\d{2})\s+(?P<desc>.*?)\s+(?P<bal>\-?[\d,.]+)\\n',
            'expected_results': ['date', 'desc', 'amt']
        }
    }

    df = pd.DataFrame()
    try:
        _rs_dct = {}
        for k, v in fields.items():
            find = re.findall(v['regex'], page_content)
            if len(find) > 0:
                for item in find:
                    dct = dict(
                    zip(list(map(lambda x: f'{k}_{x}', v['expected_results'])), item))
                    df = pd.concat([df, pd.DataFrame([dct])], ignore_index=True) 
    except Exception as e:
        logger.info(f"Couldn't read transaction line item: {e}")
        return pd.DataFrame()

    # float columns
    convert_cols = ['transaction_amt'] # there's only one float column in the card statement...
    df.loc[:, df.columns.isin(convert_cols)] = df.loc[:, df.columns.isin(convert_cols)].replace('[\,\$\s]', '',
                                                                                            regex=True).fillna(0).astype(float)
    # flip sign for card transactions - negative is credit and positive is charge
    df.loc[:, df.columns.isin(convert_cols)] = df.loc[:, df.columns.isin(convert_cols)].mul(-1)

    '''# clean up column names
    _l = list(map(lambda x: x.replace("line_item_", ''), df.columns.to_list()))
    df.rename(columns=dict(zip(df.columns, _l)), inplace=True)'''  

    # rename columns to match database schema
    df = df.rename(columns={
        'transaction_date': 'date',
        'transaction_desc': 'desc'
    })

    # add res_balance column to match database schema
    df['res_balance'] = np.nan
    df['res_balance'] = df['res_balance'].astype(float)

    # sort by date -> format as datetime first
    df['adjusted_date'] = df['date'].apply(
    lambda x: f"{current_year - 1}/{x}" if x.startswith('12/') else f"{current_year}/{x}"
    )
    df['adjusted_date'] = pd.to_datetime(df['adjusted_date'], format='%Y/%m/%d')
    df = df.sort_values(by='adjusted_date').reset_index(drop=True)

    # Drop the adjusted_date column if no longer needed
    # df.drop(columns=['adjusted_date'], inplace=True)

    return df

def extract_invoice_data(file_path):
    logger.info(f"Processing file {file_path}")
    reader = PdfReader(file_path)
    total_pages = reader.get_num_pages()
    page_no = 1

    # determine invoice type
    invoice_type = get_inv_type(reader)
    if invoice_type == 'unknown':
        logger.info("Unexpected invoice type, formatting of invoice may have changed.")
        return

    logger.info(f"Reading invoice:")
    line_items = pd.DataFrame()
    bank_summary, card_summary = pd.DataFrame(), pd.DataFrame()
    page_content = get_page_content(page_no, total_pages, reader, invoice_type)
    if invoice_type == 'bank':
        line_items = get_bank_line_items(page_content)
        bank_summary = get_bank_summary_details(reader)  # always first page
        line_items['invoice_id'] = bank_summary['invoice_id'].values[0] if not bank_summary.empty else None
    elif invoice_type == 'card':
        line_items = get_card_line_items(page_content)
        card_summary = get_card_summary_details(reader)
        line_items['invoice_id'] = card_summary['invoice_id'].values[0] if not card_summary.empty else None
    else:
        logger.info("Unknown invoice type, unable to parse line items.")
        return

    # Log summary and line items data
    if not bank_summary.empty:
        logger.info("\nSummary Data Table:")
        logger.info("\n" + bank_summary.to_string(index=False))

    if not card_summary.empty:
        logger.info("\nSummary Data Table:")
        logger.info("\n" + card_summary.to_string(index=False))
    
    if not line_items.empty:
        logger.info(f"\nTransaction Line Items Table ({len(line_items)} records):")
        logger.info("\n" + line_items.to_string(index=True))

    # save parsed data to sqlite db
    conn = sqlite3.connect('finance_data.db')
    cursor = conn.cursor()
    success = False  # Track if database operations succeed
    try:
        if not bank_summary.empty:
            # Convert values to proper types before insert
            values = (
                str(bank_summary['invoice_id'].iloc[0]),
                float(bank_summary['beginning_balance'].iloc[0]),
                float(bank_summary['ending_balance'].iloc[0]),
                float(bank_summary['deposits'].iloc[0]),
                float(bank_summary['withdrawals'].iloc[0]),
                bank_summary['date_start'].iloc[0].strftime('%Y-%m-%d'),
                bank_summary['date_end'].iloc[0].strftime('%Y-%m-%d'),
                str(bank_summary['account_number'].iloc[0]),
                str(bank_summary['invoice_id'].iloc[0])  # for COALESCE
            )
            
            cursor.execute('''
                INSERT INTO bank_summary 
                (invoice_id, beginning_balance, ending_balance, deposits, withdrawals, 
                 date_start, date_end, account_number, created_on, modified_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT created_on FROM bank_summary WHERE invoice_id = ?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP)
                ON CONFLICT(invoice_id) DO UPDATE SET 
                    beginning_balance=excluded.beginning_balance,
                    ending_balance=excluded.ending_balance,
                    deposits=excluded.deposits,
                    withdrawals=excluded.withdrawals,
                    date_start=excluded.date_start,
                    date_end=excluded.date_end,
                    account_number=excluded.account_number,
                    modified_on=CURRENT_TIMESTAMP''', values)
            
        if not card_summary.empty:
            # Convert values to proper types before insert
            values = (
                str(card_summary['invoice_id'].iloc[0]),
                str(card_summary['card_number'].iloc[0]),
                float(card_summary['previous_balance'].iloc[0]),
                float(card_summary['current_balance'].iloc[0]),
                card_summary['date_start'].iloc[0].strftime('%Y-%m-%d'),
                card_summary['date_end'].iloc[0].strftime('%Y-%m-%d'),
                float(card_summary['cash_advances'].iloc[0]),
                float(card_summary['balance_transfers'].iloc[0]),
                float(card_summary['fees'].iloc[0]),
                float(card_summary['interest'].iloc[0]),
                float(card_summary['available_credit'].iloc[0]),
                float(card_summary['credit_limit'].iloc[0]),
                str(card_summary['invoice_id'].iloc[0]) # for COALESCE
            )
            
            cursor.execute('''
                INSERT INTO card_summary 
                (invoice_id, card_number, previous_balance, current_balance, date_start, 
                 date_end, cash_advances, balance_transfers, fees, interest, available_credit, credit_limit,
                 created_on, modified_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT created_on FROM card_summary WHERE invoice_id = ?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP)
                ON CONFLICT(invoice_id) DO UPDATE SET 
                    card_number=excluded.card_number,
                    previous_balance=excluded.previous_balance,
                    current_balance=excluded.current_balance,
                    date_start=excluded.date_start,
                    date_end=excluded.date_end,
                    cash_advances=excluded.cash_advances,
                    balance_transfers=excluded.balance_transfers,
                    fees=excluded.fees,
                    interest=excluded.interest,
                    available_credit=excluded.available_credit,
                    credit_limit=excluded.credit_limit,
                    modified_on=CURRENT_TIMESTAMP''', values)

        '''if not line_items.empty:
            # First delete existing transactions for this invoice_id
            cursor.execute('DELETE FROM transactions WHERE invoice_id = ?', 
                         (line_items['invoice_id'].iloc[0],))
            
            # Add timestamps and insert new transactions
            line_items['modified_on'] = datetime.now()
            line_items.to_sql('transactions', conn, if_exists='append', index=False)'''
        
        if not line_items.empty:
            # First delete existing transactions but preserve created_on dates
            existing_created = {}
            cursor.execute('''
                SELECT invoice_id, date, desc, created_on 
                FROM transactions 
                WHERE invoice_id = ?
            ''', (line_items['invoice_id'].iloc[0],))
            
            for row in cursor.fetchall():
                key = (row[0], row[1], row[2])
                existing_created[key] = row[3]

            # Delete existing transactions
            cursor.execute('DELETE FROM transactions WHERE invoice_id = ?', 
                        (line_items['invoice_id'].iloc[0],))
            
            # Prepare transactions with proper timestamps
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            values = []
            for _, row in line_items.iterrows():
                key = (row['invoice_id'], row['date'], row['desc'])
                # Convert adjusted_date to string format
                adjusted_date_str = row['adjusted_date'].strftime('%Y-%m-%d') if pd.notnull(row['adjusted_date']) else None
                values.append((
                    row['invoice_id'],
                    row['date'],
                    row['desc'],
                    float(row['transaction_amt']),  # Ensure float type
                    float(row['res_balance']) if pd.notnull(row['res_balance']) else None,  # Handle NaN
                    adjusted_date_str,  # Use string format
                    existing_created.get(key, current_time),
                    current_time
                ))
            
            # Insert with preserved created_on dates
            cursor.executemany('''
                INSERT INTO transactions 
                (invoice_id, date, desc, transaction_amt, res_balance, adjusted_date, 
                created_on, modified_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)
            
        conn.commit()
        success = True  # Mark as successful if we get here
    except sqlite3.IntegrityError:
        # Handle duplicate invoice_id in summary table
        if not bank_summary.empty:
            bank_summary.to_sql('bank_summary', conn, if_exists='replace', index=False)
        if not card_summary.empty:
            card_summary.to_sql('card_summary', conn, if_exists='replace', index=False)
        conn.commit()
        success = True  # Mark as successful after handling duplicate
    except Exception as e:
        logger.info(f"Error saving data to database: {e}")
        conn.rollback()
        success = False  # Ensure failure is marked
    finally:
        conn.close()

    # Only archive file if database operations succeeded
    if success:
        try:
            dst_path = f"invoice_archive/{invoice_type}/{os.path.basename(file_path)}"
            logger.info(f"Moving file to {dst_path}")
            shutil.move(os.path.abspath(file_path), dst_path)
        except Exception as e:
            logger.info(f"Unable to archive file because: {e}")
    else:
        logger.info(f"File {file_path} not archived due to database errors")

def update_table_schema(file_path: str):
    """Add category column if it doesn't exist.
        Initially created db without category column; now using llm to generate.
    """
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # Check if category column exists
    cursor.execute("PRAGMA table_info(transactions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'category' not in columns:
        cursor.execute('ALTER TABLE transactions ADD COLUMN category TEXT')
        conn.commit()
    
    conn.close()

def get_uncategorized_transactions() -> pd.DataFrame:
    """Get transactions without categories, use id to update later"""
    conn = sqlite3.connect('finance_data.db')
    query = """
        SELECT id, desc
        FROM transactions 
        WHERE category IS NULL 
        AND desc IS NOT NULL 
        AND desc != ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_categories() -> List[str]:
    """Define allowed categories"""
    return [
        "Groceries",
        "Dining",
        "Transportation",
        "Utilities",
        "Entertainment",
        "Shopping",
        "Healthcare",
        "Travel",
        "Income",
        "Transfer",
        "Other"
    ]

def categorize_descriptions(descriptions: pd.DataFrame, categories: List[str]) -> Dict[str, str]:
    """Use local LLM to categorize descriptions"""
    categorized = {}
    total = len(descriptions)
    
    for idx, row in descriptions.iterrows():
        try:
            # Get category from LLM
            category = categorize_transaction(row['desc'], categories)
            
            # Validate and store result with transaction ID
            if category.strip() in categories:
                categorized[row['id']] = category.strip()
            else:
                categorized[row['id']] = "Other"
                
        except Exception as e:
            print(f"Error categorizing '{row['desc']}' (ID: {row['id']}): {e}")
            categorized[row['id']] = "Other"
    
    return categorized

def save_categories(categorized: Dict[str, str]):
    """Update database with new categories"""
    conn = sqlite3.connect('finance_data.db')
    cursor = conn.cursor()
    
    # Update categories by ID
    cursor.executemany("""
        UPDATE transactions 
        SET category = ?,
            modified_on = CURRENT_TIMESTAMP
        WHERE id = ?
    """, [(category, id) for id, category in categorized.items()])
    
    conn.commit()
    conn.close()

def review_categories(categorized: Dict[str, str]) -> pd.DataFrame:
    """Convert results to DataFrame for review with descriptions"""
    conn = sqlite3.connect('finance_data.db')
    
    # Get descriptions for the IDs
    id_list = list(categorized.keys())
    placeholders = ','.join('?' * len(id_list))
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, desc FROM transactions WHERE id IN ({placeholders})", id_list)
    descriptions = dict(cursor.fetchall())
    
    # Create review DataFrame with both description and category
    df = pd.DataFrame({
        'description': [descriptions[id] for id in categorized.keys()],
        'category': list(categorized.values())
    })
    
    conn.close()
    return df

def main():
    create_tables() # create tables if they don't exist
    #files = ['invoices/card/20250202-statements-0907-.pdf']
    #files = ['invoices/bank/20250117-statements-3923-.pdf']

    files = glob.glob('invoices/**/*.pdf')
    for file_path in files:
        extract_invoice_data(file_path) 

    update_table_schema('finance_data.db')

    # Get uncategorized transactions
    descriptions = get_uncategorized_transactions()
    if descriptions.empty:
        print("No uncategorized transactions found.")
        return
        
    categories = get_categories()
    
    # Categorize only new descriptions
    categorized = categorize_descriptions(descriptions, categories)
    
    # Create DataFrame for review
    results = review_categories(categorized)
    print("\nProposed categories for review:")
    logger.info(results)
    
    # Ask for confirmation
    response = input("\nDo you want to save these categories? (y/n): ")
    if response.lower() == 'y':
        save_categories(categorized)
        print("Categories saved to database.")
    else:
        print("Categories not saved.")
        results.to_csv('pending_categories.csv')
        print("Categories exported to pending_categories.csv for manual review.")

if __name__ == '__main__':
    logger.info(f"======================== Version \t{__version__}\t========================")
    main()