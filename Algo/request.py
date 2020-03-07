import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'news':  (0, 8691)	0.6929059804139004
  (0, 2170)	0.7210279483533572})

print(r.json())