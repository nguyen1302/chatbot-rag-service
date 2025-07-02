import argparse
from app.services.ingest_service import recreate_collection_if_needed, ingest_single_file, ingest_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Ingest single file")
    parser.add_argument("--id", help="Document ID (used if --file is set)")
    parser.add_argument("--dir", help="Folder containing PDFs (default = raw-data)", default="raw-data")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if exists")
    args = parser.parse_args()

    if args.file:
        if not args.id:
            raise ValueError("If using --file, you must also provide --id")
        recreate_collection_if_needed()
        ingest_single_file(args.file, args.id, args.force)
    else:
        ingest_folder(args.dir, args.force)

    """
    Examples:
    python scripts/ingest_file.py
    python scripts/ingest_file.py --force
    python scripts/ingest_file.py --file raw-data/bktech.pdf --id bktech
    """
