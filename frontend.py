import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from backend import StockPredictor
import threading
import os

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction App")
        self.root.geometry("1200x800")
        self.api_key = "23710d40f10cfd625895b108df18971d"  # Replace with your MarketStack key
        self.predictor = StockPredictor(self.api_key)
        
        # UI Setup
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0')
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        self.control_frame = ttk.Frame(self.main_frame, width=300, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Visualization Panel
        self.visualization_frame = ttk.Frame(self.main_frame, padding=10)
        self.visualization_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel()
        self.create_visualization_panel()
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")
    
    def create_control_panel(self):
        """Create input controls"""
        # Search Frame
        search_frame = ttk.Frame(self.control_frame)
        search_frame.pack(fill=tk.X, pady=5)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(search_frame, text="🔍", width=3, 
                  command=self.search_company).pack(side=tk.RIGHT)
        
        # Symbol Input
        symbol_frame = ttk.Frame(self.control_frame)
        symbol_frame.pack(fill=tk.X, pady=5)
        ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(symbol_frame)
        self.symbol_entry.pack(fill=tk.X)
        self.symbol_entry.insert(0, "AAPL")
        
        # Date Range
        date_frame = ttk.Frame(self.control_frame)
        date_frame.pack(fill=tk.X, pady=5)
        ttk.Label(date_frame, text="Start:").grid(row=0, column=0, sticky='w')
        self.start_date = ttk.Entry(date_frame)
        self.start_date.grid(row=0, column=1, sticky='ew')
        self.start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        
        ttk.Label(date_frame, text="End:").grid(row=1, column=0, sticky='w')
        self.end_date = ttk.Entry(date_frame)
        self.end_date.grid(row=1, column=1, sticky='ew')
        self.end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        date_frame.columnconfigure(1, weight=1)
        
        # Fetch Button
        ttk.Button(self.control_frame, text="Fetch Data", 
                  command=self.fetch_data).pack(pady=10, fill=tk.X)
        
        # Prediction Controls
        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=5)
        ttk.Label(self.control_frame, text="Prediction Days:").pack(anchor='w')
        self.pred_days = ttk.Entry(self.control_frame)
        self.pred_days.pack(fill=tk.X)
        self.pred_days.insert(0, "30")
        
        ttk.Button(self.control_frame, text="Predict", 
                  command=self.predict_prices).pack(pady=5, fill=tk.X)
        
        # Model Controls
        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Train Model", 
                  command=self.train_model).pack(pady=5, fill=tk.X)
    
    def create_visualization_panel(self):
        """Create chart and data table"""
        # Chart
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zoom/pan
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.visualization_frame)
        self.toolbar.update()
        
        # Data Table
        self.tree = ttk.Treeview(self.visualization_frame, 
                                columns=('Date','Open','High','Low','Close','Volume'), 
                                show='headings', height=8)
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.X, pady=5)
        
        # Scrollbars
        vsb = ttk.Scrollbar(self.visualization_frame, orient='vertical', 
                           command=self.tree.yview)
        hsb = ttk.Scrollbar(self.visualization_frame, orient='horizontal', 
                           command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def search_company(self):
        query = self.search_entry.get()
        if not query:
            messagebox.showwarning("Warning", "Enter a company name")
            return
        
        def search():
            results, error = self.predictor.search_company(query)
            if error:
                self.root.after(0, lambda: messagebox.showerror("Error", error))
                return
            
            if not results:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No results"))
                return
            
            popup = tk.Toplevel()
            popup.title(f"Results for '{query}'")
            tree = ttk.Treeview(popup, columns=('Symbol','Name','Exchange'))
            for col in tree['columns']:
                tree.heading(col, text=col)
            
            for res in results:
                tree.insert('', 'end', values=(res['symbol'], res['name'], res['stock_exchange']))
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            def on_select(event):
                item = tree.item(tree.focus())
                self.symbol_entry.delete(0, tk.END)
                self.symbol_entry.insert(0, item['values'][0])
                popup.destroy()
            
            tree.bind('<Double-1>', on_select)
        
        threading.Thread(target=search, daemon=True).start()
    
    def fetch_data(self):
        symbol = self.symbol_entry.get().strip()
        start = self.start_date.get().strip()
        end = self.end_date.get().strip()
        
        if not symbol:
            messagebox.showerror("Error", "Enter a stock symbol")
            return
        
        def fetch():
            self.stock_data, error = self.predictor.get_stock_data(symbol, start, end)
            if error:
                self.root.after(0, lambda: messagebox.showerror("Error", error))
                return
            
            self.root.after(0, self.update_display)
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def update_display(self):
        # Update Table
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        df = self.stock_data.reset_index()
        for _, row in df.iterrows():
            self.tree.insert('', 'end', values=(
                row['date'].strftime('%Y-%m-%d'),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{int(row['volume']):,}"
            ))
        
        # Update Chart
        self.ax.clear()
        self.ax.plot(df['date'], df['close'], label='Close Price')
        self.ax.set_title(f"{self.symbol_entry.get()} Stock Price")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price ($)")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.autofmt_xdate()
        self.canvas.draw()
    
    def train_model(self):
        if not hasattr(self, 'stock_data'):
            messagebox.showerror("Error", "Fetch data first")
            return
        
        def train():
            symbol = self.symbol_entry.get().strip()
            model, mse = self.predictor.train_model(symbol, self.stock_data)
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"Model trained (MSE: {mse:.4f})"
            ))
        
        threading.Thread(target=train, daemon=True).start()
    
    def predict_prices(self):
        if not hasattr(self, 'stock_data'):
            messagebox.showerror("Error", "Fetch data first")
            return
        
        try:
            days = int(self.pred_days.get())
        except ValueError:
            messagebox.showerror("Error", "Enter valid days")
            return
        
        def predict():
            symbol = self.symbol_entry.get().strip()
            preds, error = self.predictor.predict_future_prices(symbol, days, self.stock_data)
            if error:
                self.root.after(0, lambda: messagebox.showerror("Error", error))
                return
            
            self.root.after(0, lambda: self.show_predictions(preds))
        
        threading.Thread(target=predict, daemon=True).start()
    
    def show_predictions(self, preds):
        self.ax.plot(preds.index, preds.values, 'r--', label='Prediction')
        self.ax.legend()
        self.canvas.draw()
        
        last_price = self.stock_data['close'].iloc[-1]
        change_pct = (preds.iloc[-1] - last_price) / last_price * 100
        messagebox.showinfo(
            "Prediction", 
            f"Predicted {len(preds)} days ahead:\n"
            f"Price: ${preds.iloc[-1]:.2f} ({change_pct:.2f}% change)"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()