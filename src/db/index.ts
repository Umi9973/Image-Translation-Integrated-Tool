import Dexie, { type EntityTable } from 'dexie'
import type { Project, MangaPage } from '../types'

const db = new Dexie('MangaVibeDB') as Dexie & {
  projects: EntityTable<Project, 'id'>
  pages: EntityTable<MangaPage, 'id'>
}

db.version(1).stores({
  projects: 'id, name, createdAt',
  pages: 'id, projectId, filename, createdAt',
})

export { db }
