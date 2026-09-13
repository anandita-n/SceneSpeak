"""Run the dev server with the data/ directory excluded from the
auto-reloader. Without this, uvicorn's --reload watches the whole project
folder, and every board generation (which writes to data/scenespeak.db
and data/chroma/) triggers a full server restart mid-request -- which
shows up in the browser as a bare "Failed to fetch" with no explanation.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        reload=True,
        reload_excludes=["data/*", "*.db"],
    )
