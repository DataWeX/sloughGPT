import AsyncStorage from '@react-native-async-storage/async-storage'
import * as folders from '../folders'

const STORAGE_KEY = '@sloughgpt/session-folders'

beforeEach(async () => {
  jest.clearAllMocks()
  await AsyncStorage.clear()
  folders._resetCache()
})

describe('folders', () => {
  describe('getFolders', () => {
    it('returns empty array when no folders exist', async () => {
      expect(await folders.getFolders()).toEqual([])
    })

    it('returns stored folders', async () => {
      await AsyncStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          folders: [{ id: 'f1', name: 'Work', createdAt: 1 }],
          assignments: {},
        }),
      )
      folders._resetCache()
      const result = await folders.getFolders()
      expect(result).toEqual([{ id: 'f1', name: 'Work', createdAt: 1 }])
    })
  })

  describe('createFolder', () => {
    it('creates a folder with a unique id', async () => {
      const folder = await folders.createFolder('Work')
      expect(folder.name).toBe('Work')
      expect(folder.id).toBeTruthy()
      const all = await folders.getFolders()
      expect(all).toHaveLength(1)
      expect(all[0].name).toBe('Work')
    })

    it('rejects empty names', async () => {
      await expect(folders.createFolder('   ')).rejects.toThrow()
      expect(await folders.getFolders()).toEqual([])
    })

    it('rejects duplicate names case-insensitively', async () => {
      await folders.createFolder('Work')
      await expect(folders.createFolder('work')).rejects.toThrow()
      expect(await folders.getFolders()).toHaveLength(1)
    })
  })

  describe('renameFolder', () => {
    it('renames an existing folder', async () => {
      const f = await folders.createFolder('Work')
      await folders.renameFolder(f.id, 'Office')
      const all = await folders.getFolders()
      expect(all[0].name).toBe('Office')
    })

    it('throws for unknown folder id', async () => {
      await expect(folders.renameFolder('nope', 'X')).rejects.toThrow()
    })
  })

  describe('deleteFolder', () => {
    it('removes folder and clears assignments pointing at it', async () => {
      const f = await folders.createFolder('Work')
      await folders.moveSessionToFolder('s1', f.id)
      await folders.deleteFolder(f.id)
      expect(await folders.getFolders()).toEqual([])
      expect(await folders.getFolderOf('s1')).toBeNull()
    })
  })

  describe('moveSessionToFolder', () => {
    it('assigns a session to a folder', async () => {
      const f = await folders.createFolder('Work')
      await folders.moveSessionToFolder('s1', f.id)
      expect(await folders.getFolderOf('s1')).toBe(f.id)
    })

    it('moves a session between folders', async () => {
      const a = await folders.createFolder('A')
      const b = await folders.createFolder('B')
      await folders.moveSessionToFolder('s1', a.id)
      await folders.moveSessionToFolder('s1', b.id)
      expect(await folders.getFolderOf('s1')).toBe(b.id)
    })

    it('unfiles a session with null', async () => {
      const f = await folders.createFolder('Work')
      await folders.moveSessionToFolder('s1', f.id)
      await folders.moveSessionToFolder('s1', null)
      expect(await folders.getFolderOf('s1')).toBeNull()
    })

    it('throws when assigning to an unknown folder', async () => {
      await expect(folders.moveSessionToFolder('s1', 'ghost')).rejects.toThrow()
    })
  })

  describe('getSessionsInFolder', () => {
    it('returns assigned session ids only', async () => {
      const f = await folders.createFolder('Work')
      await folders.moveSessionToFolder('s1', f.id)
      await folders.moveSessionToFolder('s2', f.id)
      await folders.moveSessionToFolder('s3', null)
      const ids = await folders.getSessionsInFolder(f.id)
      expect(ids.sort()).toEqual(['s1', 's2'])
    })
  })
})
