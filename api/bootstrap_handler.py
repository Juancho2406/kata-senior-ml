def handler(event, context):
    del context
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": '{"message":"bootstrap image"}',
    }
