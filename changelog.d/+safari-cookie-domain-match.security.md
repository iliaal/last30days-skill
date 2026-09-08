Safari cookie extraction now matches the cookie host exactly or as a true subdomain, so a session cookie stored for an unrelated host such as `x.com.evil.tld` is no longer picked up for `x.com`.
