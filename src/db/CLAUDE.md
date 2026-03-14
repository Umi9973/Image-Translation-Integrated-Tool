# DB Module

Dexie.js wrapper around browser IndexedDB. All persistence goes through here.

## File
`index.ts` — exports a single `db` instance. Import it wherever you need persistence.

```typescript
import { db } from '../db'
```

## Schema (v1)
| Table | Primary Key | Indexed fields | Stores |
|---|---|---|---|
| `projects` | `id` | `name`, `createdAt` | `Project` objects |
| `pages` | `id` | `projectId`, `filename`, `createdAt` | `MangaPage` objects (incl. bubbles array + imageBlob) |

## Rules
- Never define new Dexie tables outside of `index.ts`. If schema changes are needed, bump the version number and add a migration block.
- `imageBlob` is stored directly in IndexedDB for moderate-size images. For large image caches, use OPFS instead and store only the OPFS file handle reference in the DB.
- All DB calls are async — always `await` them. Never fire-and-forget a write.
- Bubble arrays are stored as a JSON column inside `MangaPage`, not as a separate table. Keep it simple until query needs demand otherwise.
