# https://pythonprogramming.net/data-visualization-application-dash-python-tutorial-introduction/
# Intro - Data Visualization Applications with Dash and Python p.1

import dash
import dash_core_components as dcc
import dash_html_components as html

# dash is a flask app
app = dash.Dash()

# data = a dictionary or a list of dictionaries
app.layout = html.Div(children=[
    html.H1('Dash tutorials'),
    dcc.Graph(id='example', figure={
        'data': [{'x': [1, 2, 3, 4, 5], 'y':[5, 6, 7, 8, 9], 'type':'line', 'name':'boats'},
                 {'x': [8, 2, 4, 1, 10], 'y':[9, 5, 2, 12, 1],
                     'type':'bar', 'name':'cars'},
                 ],
        'layout': {
            'title': 'Basic Dash Example'
        }
    })
])

if __name__ == '__main__':
    app.run_server(debug=True)
