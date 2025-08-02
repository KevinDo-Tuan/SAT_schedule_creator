from datetime import datetime

def format_date(value, format='%B %d, %Y'):
    """Format a date string to a more readable format"""
    if not value:
        return ""
    try:
        # Try to parse the date if it's a string
        if isinstance(value, str):
            # Handle different possible date formats
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
                try:
                    value = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
        # If we have a datetime object, format it
        if isinstance(value, datetime):
            return value.strftime(format)
        return str(value)
    except Exception:
        # If any error occurs, return the original value
        return str(value)

def format_number(value, precision=2):
    """Format a number with specified decimal places"""
    try:
        num = float(value)
        return f"{num:,.{precision}f}"
    except (ValueError, TypeError):
        return str(value)

def to_json(value):
    """Convert value to JSON string"""
    import json
    return json.dumps(value, ensure_ascii=False)

# Register all filters
def register_template_filters(app):
    """Register all template filters with the Flask app"""
    app.jinja_env.filters['format_date'] = format_date
    app.jinja_env.filters['format_number'] = format_number
    app.jinja_env.filters['to_json'] = to_json
