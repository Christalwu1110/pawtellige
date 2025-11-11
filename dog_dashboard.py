import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import json
import os
import traceback
import webbrowser
import time

# --- Configuration ---
LOG_PATH = "/home/yeling/dog_project/dog_monitor_log.json"
STATIC_IMAGE_PATH = "/home/yeling/dog_project/key_frames"
UPDATE_INTERVAL = 100  # Refresh every 100 ms (adjust for 10 frames per refresh, assuming frame rate)
ASSETS_DIR = "assets"
PORT = 8060
SIMULATION_DURATION = 50  # 50 seconds to simulate one day's changes
UPDATES_PER_SIMULATION = int(SIMULATION_DURATION * 1000 / UPDATE_INTERVAL)  # Number of updates in 50 seconds

# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# --- Ensure Assets Directory ---
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
if os.path.exists(os.path.join(ASSETS_DIR, "key_frames")):
    if not os.path.islink(os.path.join(ASSETS_DIR, "key_frames")) or not os.path.exists(os.readlink(os.path.join(ASSETS_DIR, "key_frames"))):
        os.remove(os.path.join(ASSETS_DIR, "key_frames"))
if not os.path.exists(os.path.join(ASSETS_DIR, "key_frames")):
    try:
        os.symlink(STATIC_IMAGE_PATH, os.path.join(ASSETS_DIR, "key_frames"))
        print(f"Symlinked {STATIC_IMAGE_PATH} to {ASSETS_DIR}/key_frames")
    except Exception as e:
        print(f"Error creating symlink: {e}")

# --- Layout ---
app.layout = html.Div([
    html.Div([
        html.H1("Dog Behavior Monitoring Dashboard", className='white-tit'),
        html.Span(id='current-time', className='white-tit time')
    ], className='head-glass clearfix'),
    html.Div([
        html.Div([
            html.Div([
                html.H3("Pose Distribution", className='tit-text'),
            ], className='box-tit-glass'),
            dcc.Graph(id='pose-pie-chart', style={'width': '80%', 'height': '300px', 'margin': '0 auto'}),
            html.Div([
                html.H3("Pose Timeline", className='tit-text'),
            ], className='box-tit-glass'),
            dcc.Graph(id='pose-timeline', style={'width': '90%', 'height': '300px', 'margin': '0 auto'}),
        ], className='chart-box', style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H3("Total Records", className='kpi-label'),
                    html.Div(id='total-records', className='kpi-value'),
                ], className='kpi-glass-card', style={'width': '33%', 'margin-right': '1%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Alert Count", className='kpi-label'),
                    html.Div(id='alert-count', className='kpi-value'),
                ], className='kpi-glass-card', style={'width': '33%', 'margin-right': '1%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Alert Percentage", className='kpi-label'),
                    html.Div(id='alert-percentage', className='kpi-value'),
                ], className='kpi-glass-card', style={'width': '33%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
            html.Div([
                html.Div([
                    html.H3("Latest Frame", className='tit-text'),
                ], className='box-tit-glass'),
                html.Img(id='latest-frame', style={'width': '100%', 'max-width': '400px', 'margin': 'auto', 'display': 'block'}),
            ], className='chart-box'),
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Div([
                html.H3("Confidence Distribution", className='tit-text'),
            ], className='box-tit-glass'),
            dcc.Graph(id='confidence-histogram', style={'width': '80%', 'height': '300px', 'margin': '0 auto'}),
            html.Div([
                html.H3("Abnormal Behavior Alerts", className='tit-text'),
            ], className='box-tit-glass'),
            html.Div([
                html.Table(id='alert-table', style={'width': '100%', 'border-collapse': 'collapse'}),
                html.Button("Show More", id='toggle-alerts', n_clicks=0, style={
                    'margin-top': '10px',
                    'padding': '8px 16px',
                    'background-color': '#4ade80',
                    'color': '#fff',
                    'border': 'none',
                    'border-radius': '4px',
                    'cursor': 'pointer'
                }),
                html.Div(id='extra-alerts', style={'display': 'none'})
            ], className='dash-table-container', style={'width': '95%', 'margin': '0 auto', 'overflow-x': 'auto'}),
        ], className='chart-box', style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], className='mainbox'),
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL, n_intervals=0),
    dcc.Interval(id='time-update', interval=1000, n_intervals=0)
], style={'min-height': '100vh'})

# --- Load Log Data ---
def load_log_data():
    for _ in range(3):
        try:
            if not os.path.exists(LOG_PATH):
                print(f"Log file not found: {LOG_PATH}")
                return pd.DataFrame()
            
            print(f"Attempting to load log file: {LOG_PATH}")
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                print("Log file is empty")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            required_columns = ['frame', 'path', 'predicted_pose', 'confidence', 'timestamp', 'abnormal_flags']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in log: {missing_columns}")
                return pd.DataFrame()
            
            df['abnormal_flags'] = df['abnormal_flags'].apply(parse_abnormal_flags)
            df = df.drop_duplicates(subset=['frame'], keep='last').reset_index(drop=True)
            
            print(f"Loaded {len(df)} records from {LOG_PATH} after deduplication")
            if len(df) > 0:
                print(f"Sample record: {df.iloc[-1].to_dict()}")
            return df
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading log: {e}")
            time.sleep(0.2)
    return pd.DataFrame()

# --- Parse Abnormal Flags ---
def parse_abnormal_flags(x):
    try:
        if isinstance(x, str) and x.startswith('['):
            return json.loads(x)
        elif isinstance(x, list):
            return x
        else:
            return []
    except (json.JSONDecodeError, TypeType) as e:
        print(f"Error parsing abnormal_flags: {e}, value: {x}")
        return []

# --- Callback for Updating Time ---
@app.callback(
    Output('current-time', 'children'),
    Input('time-update', 'n_intervals')
)
def update_time(n_intervals):
    return f"Time: {time.strftime('%H:%M %Z, %Y-%m-%d')}"

# --- Callback for Updating Dashboard ---
@app.callback(
    [Output('pose-pie-chart', 'figure'),
     Output('pose-timeline', 'figure'),
     Output('total-records', 'children'),
     Output('alert-count', 'children'),
     Output('alert-percentage', 'children'),
     Output('confidence-histogram', 'figure'),
     Output('alert-table', 'children'),
     Output('extra-alerts', 'children'),
     Output('latest-frame', 'src')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n_intervals):
    try:
        full_df = load_log_data()
        
        if full_df.empty:
            print("No data to display in dashboard")
            return (
                go.Figure(),
                go.Figure(),
                "0",
                "0",
                "0%",
                go.Figure(),
                html.P("No alerts available", style={'text-align': 'center', 'color': '#fff'}),
                html.P(""),
                ''
            )
        
        # Simulate progressive data loading over 50 seconds
        total_records_full = len(full_df)
        step = max(1, total_records_full // UPDATES_PER_SIMULATION)  # Data points per update
        current_step = (n_intervals % UPDATES_PER_SIMULATION) + 1
        df = full_df.iloc[:current_step * step]
        
        # Reset simulation if end reached (loop)
        if current_step >= UPDATES_PER_SIMULATION:
            # Optional: Reset n_intervals or loop
            pass
        
        pose_counts = df['predicted_pose'].value_counts()
        total = len(df)
        print(f"Pose counts: {pose_counts.to_dict()}")
        pie_fig = go.Figure(data=[
            go.Pie(labels=pose_counts.index, values=pose_counts.values, hole=0.3, marker_colors=['#4c51bf', '#63b3ed', '#38a169', '#ed8936', '#9f7aea'])
        ])
        pie_fig.update_layout(
            title="Pose Distribution",
            title_x=0.5,
            title_font=dict(size=16, family='Montserrat, "Microsoft YaHei", Arial, sans-serif', color='#fff', weight='bold'),
            legend=dict(font=dict(size=14, color='#fff'), orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1, bgcolor='rgba(0,0,0,0.5)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=0, r=50),
            font=dict(family='Montserrat, "Microsoft YaHei", Arial, sans-serif', size=14, color='#fff'),
            showlegend=True
        )
        
        timeline_fig = go.Figure(data=[
            go.Scatter(
                x=df['timestamp'],
                y=df['predicted_pose'],
                mode='lines+markers',
                marker=dict(size=8, color='#4c51bf'),
                line=dict(color='#63b3ed')
            )
        ])
        timeline_fig.update_layout(
            title="Pose Timeline",
            title_x=0.5,
            title_font=dict(size=16, family='Montserrat, "Microsoft YaHei", Arial, sans-serif', color='#fff', weight='bold'),
            xaxis_title="Time",
            yaxis_title="Pose",
            xaxis=dict(tickfont=dict(size=12, color='#fff'), showticklabels=True, tickangle=45),
            yaxis=dict(tickfont=dict(size=12, color='#fff'), showticklabels=True),
            legend=dict(font=dict(size=14, color='#fff'), bgcolor='rgba(0,0,0,0.5)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=0, r=0),
            font=dict(family='Montserrat, "Microsoft YaHei", Arial, sans-serif', size=14, color='#fff'),
            showlegend=True
        )
        
        total_records = str(len(df))
        alerts = df[df['abnormal_flags'].apply(lambda x: len(x) > 0)]
        alert_count = str(len(alerts))
        print(f"Alerts found: {len(alerts)}")
        alert_percentage = f"{(len(alerts) / total * 100):.2f}%" if total > 0 else "0%"
        
        confidence_fig = go.Figure(data=[
            go.Histogram(
                x=df['confidence'],
                nbinsx=20,
                marker=dict(color='#4c51bf')
            )
        ])
        confidence_fig.update_layout(
            title="Confidence Distribution",
            title_x=0.5,
            title_font=dict(size=16, family='Montserrat, "Microsoft YaHei", Arial, sans-serif', color='#fff', weight='bold'),
            xaxis_title="Confidence",
            yaxis_title="Count",
            xaxis=dict(tickfont=dict(size=12, color='#fff'), showticklabels=True),
            yaxis=dict(tickfont=dict(size=12, color='#fff'), showticklabels=True),
            legend=dict(font=dict(size=14, color='#fff'), bgcolor='rgba(0,0,0,0.5)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=0, r=0),
            font=dict(family='Montserrat, "Microsoft YaHei", Arial, sans-serif', size=14, color='#fff'),
            showlegend=True
        )
        
        if not alerts.empty:
            recent_alerts = alerts.tail(3)[::-1]
            alert_table = html.Table([
                html.Thead(
                    html.Tr([html.Th(col, style={
                        'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                        'padding': '8px',
                        'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                        'color': '#4ade80',
                        'font-weight': 'bold',
                        'white-space': 'nowrap'
                    }) for col in ['Timestamp', 'Pose', 'Confidence', 'Alerts']])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['timestamp'], style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(row['predicted_pose'], style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(f"{row['confidence']:.2f}", style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(', '.join(row['abnormal_flags']), style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        })
                    ]) for index, row in recent_alerts.iterrows()
                ])
            ], style={'width': '100%', 'border-collapse': 'collapse'})
            
            extra_alerts = alerts[:-3][::-1] if len(alerts) > 3 else pd.DataFrame()
            extra_alerts_table = html.Table([
                html.Thead(
                    html.Tr([html.Th(col, style={
                        'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                        'padding': '8px',
                        'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                        'color': '#4ade80',
                        'font-weight': 'bold',
                        'white-space': 'nowrap'
                    }) for col in ['Timestamp', 'Pose', 'Confidence', 'Alerts']])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['timestamp'], style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(row['predicted_pose'], style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(f"{row['confidence']:.2f}", style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        }),
                        html.Td(', '.join(row['abnormal_flags']), style={
                            'border-bottom': '1px solid rgba(255, 255, 255, 0.3)',
                            'padding': '8px',
                            'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                            'color': '#fff',
                            'white-space': 'nowrap'
                        })
                    ]) for index, row in extra_alerts.iterrows()
                ])
            ], style={'width': '100%', 'border-collapse': 'collapse'}) if not extra_alerts.empty else html.P("")
        else:
            alert_table = html.P("No alerts available", style={
                'text-align': 'center',
                'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                'color': '#fff'
            })
            extra_alerts_table = html.P("")
        
        latest_frame_path = df['path'].iloc[-1] if not df.empty else ''
        print(f"Checking latest frame path: {latest_frame_path}")
        if os.path.exists(latest_frame_path):
            frame_url = app.get_asset_url(os.path.join('key_frames', os.path.basename(latest_frame_path)))
            print(f"Latest frame URL: {frame_url}")
        else:
            frame_url = ''
            print(f"Latest frame not found: {latest_frame_path}")
        
        return pie_fig, timeline_fig, total_records, alert_count, alert_percentage, confidence_fig, alert_table, extra_alerts_table, frame_url
    
    except Exception as e:
        print(f"Error in update_dashboard: {e}")
        print(traceback.format_exc())
        return (
            go.Figure(),
            go.Figure(),
            "0",
            "0",
            "0%",
            go.Figure(),
            html.P(f"Error: {str(e)}", style={
                'text-align': 'center',
                'font-family': 'Montserrat, "Microsoft YaHei", Arial, sans-serif',
                'color': '#e53e3e'
            }),
            html.P(""),
            ''
        )

# --- Callback for Toggling Extra Alerts ---
@app.callback(
    [Output('extra-alerts', 'style'),
     Output('toggle-alerts', 'children')],
    [Input('toggle-alerts', 'n_clicks')],
    [State('extra-alerts', 'style')]
)
def toggle_extra_alerts(n_clicks, current_style):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, "Show More"
    else:
        return {'display': 'block'}, "Show Less"

# --- Run App ---
if __name__ == '__main__':
    try:
        print(f"Starting Dash server on http://127.0.0.1:{PORT}")
        app.run_server(debug=True, host='0.0.0.0', port=PORT)
        time.sleep(2)
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception as e:
        print(f"Error starting Dash server: {e}")
        print(traceback.format_exc())
        