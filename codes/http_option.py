import http.client

conn = http.client.HTTPSConnection("api.astronomyapi.com")

payload = '{"style":{"moonStyle":"default","backgroundStyle":"stars","backgroundColor":"#000000","headingColor":"#ffffff","textColor":"#ffffff"},"observer":{"latitude":33.775867,"longitude":-84.39733,"date":"2024-08-13"},"view":{"type":"portrait-simple","parameters":{}}}'

headers = {
    "Authorization": "Basic N2M1MGZjMDctOGNjOS00YTY3LWJmNGEtZDdlODdkZDk1NmM5OmNhODRiMzUyNWUwNzZkZmQyMTJmMWU4YzBiMWQxNGZkYzllZTBmYjVlYWQ1MjhiMDQzNjQwMTNiNDg5ZmU5NWVlMWFjZjhhYjQ1NDA3ODYwYjgzNjAxZDJkMWZiMzg1ZDNhMmRhODc5YzVjMDE3YTg5NTYwOWRlNDY2YjY3YmIwMjkxZWI5NjU2MGQ1ZWQ1ZjI1Y2QyOThhZWZiY2Q0NjRkNWQ1YzY5Y2IwZDQyYmQ1OTUxMjk5NGExNmYyMDYwNDJkMWY2MTI3NTEzZDY4ZTFjZTg5ZTAxZjBlMTUyYzcw"
}

conn.request("POST", "/api/v2/studio/moon-phase", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
