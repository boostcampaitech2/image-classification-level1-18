from pytz import timezone
import datetime as dt


def get_time() -> str:
    """
    Return time
    """
    return (
        dt.datetime.now()
        .astimezone(timezone("Asia/Seoul"))
        .strftime("%Y-%m-%d_%H%M%S")
    )
