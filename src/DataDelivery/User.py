class User:

    # Possible values for __init__ method:
    # email: anything formatted as [name]@[domain].[suffix]
    # ntid: Any string. Must match the user's NTID in active directory or be left empty.
    # level: 'admin', 'change', or 'readonly', depending on what level of access you want to specify
    def __init__(self, email: str, ntid: str=None, level: str='readonly'):
        self.email = email
        self.ntid = ntid
        self.level = level
