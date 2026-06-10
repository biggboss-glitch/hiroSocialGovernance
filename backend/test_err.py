from openai import APIConnectionError
try:
    raise APIConnectionError(request=None)
except Exception as e:
    print('str(e):', str(e))
