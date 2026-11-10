/** The Artifacts page's left-hand list: Home, All artifacts, then the library's
 *  folder tree. Picking an entry decides what the page's main column shows.
 *
 *  Every entry is a drop target in the page's one DndContext, with ids of its
 *  own (`nav-drop:` / `nav-folder:`) so they never collide with the folder
 *  cards and breadcrumb that share that context. "All artifacts" drops unfile,
 *  exactly like the breadcrumb's root segment. */
import { ChevronDown, ChevronRight, FolderPlus, House, Library } from 'lucide-react'
import { DndDraggable, DndDroppable } from '../components/dnd'
import { PanelSectionHeader } from '../components/ui'
import { childFolders } from '../utils/artifactFolderTree'
import type { ArtifactFolder } from '../types'
import { FolderColorSwatches, FolderGlyph, FolderMenu, FolderNameInput } from '../components/library/LibraryTable'
import type { FolderActions, LibraryDrag } from '../components/library/LibraryTable'
import { i18nT } from '../i18n/t'

/** What the main column shows. A folder id that no longer exists is resolved
 *  to Home by the page before it reaches here. */
export type LibrarySelection = { kind: 'home' } | { kind: 'all' } | { kind: 'folder'; id: string }

const ROW = 'flex items-center gap-2 w-full min-w-0 px-2 py-1.5 rounded-md text-[13px] text-left border-none cursor-pointer transition-colors'
const ROW_IDLE = 'bg-transparent text-text hover:bg-bg-hover'
const ROW_ON = 'bg-accent-subtle text-accent font-semibold'
const ROW_DROP = 'ring-2 ring-accent/40'

function TopEntry({ label, icon, active, count, dropFolderId, onClick }: {
  label: string
  icon: React.ReactNode
  active: boolean
  count?: number
  /** Set to make the entry a drop target ('' unfiles). */
  dropFolderId?: string
  onClick: () => void
}) {
  const body = (ref: ((el: HTMLElement | null) => void) | undefined, isOver: boolean) => (
    <button
      ref={ref}
      type="button"
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={`${ROW} ${active ? ROW_ON : ROW_IDLE} ${isOver ? ROW_DROP : ''}`}
    >
      <span className="shrink-0 text-muted flex">{icon}</span>
      <span className="truncate min-w-0 flex-1">{label}</span>
      {count !== undefined && <span className="text-[11px] text-muted font-normal tabular-nums">{count}</span>}
    </button>
  )
  if (dropFolderId === undefined) return body(undefined, false)
  return (
    <DndDroppable id={`nav-drop:${dropFolderId || 'root'}`} data={{ type: 'folder-drop', folderId: dropFolderId }}>
      {({ setNodeRef, isOver }) => body(setNodeRef, isOver)}
    </DndDroppable>
  )
}

function FolderEntry({ folder, folders, depth, selectedId, expandedIds, onToggleExpand, onSelect, actions, counts }: {
  folder: ArtifactFolder
  folders: ArtifactFolder[]
  depth: number
  selectedId: string
  expandedIds: ReadonlySet<string>
  onToggleExpand: (id: string) => void
  onSelect: (sel: LibrarySelection) => void
  actions: FolderActions
  counts: ReadonlyMap<string, number>
}) {
  const children = childFolders(folders, folder.id)
  const expanded = expandedIds.has(folder.id)
  const active = selectedId === folder.id
  const renaming = actions.renamingId === folder.id
  return (
    <li>
      <DndDroppable id={`nav-drop:${folder.id}`} data={{ type: 'folder-drop', folderId: folder.id }}>
        {({ setNodeRef: setDropRef, isOver }) => (
          <DndDraggable id={`nav-folder:${folder.id}`} data={{ type: 'folder', id: folder.id, name: folder.name } satisfies LibraryDrag}>
            {({ setNodeRef: setDragRef, listeners, isDragging }) => (
              <div
                ref={(el) => { setDropRef(el); setDragRef(el) }}
                {...(renaming ? {} : listeners)}
                style={{ paddingLeft: depth * 14, opacity: isDragging ? 0.4 : 1 }}
                className={`group flex items-center rounded-md ${active ? ROW_ON : 'text-text'} ${isOver ? ROW_DROP : ''}`}
              >
                {/* The chevron's slot is kept for a leaf too, so names line up. */}
                {children.length > 0 ? (
                  <button
                    type="button"
                    onClick={() => onToggleExpand(folder.id)}
                    aria-expanded={expanded}
                    aria-label={i18nT(expanded ? 'pages.artifactsPage.collapse_folder' : 'pages.artifactsPage.expand_folder', { name: folder.name })}
                    className="shrink-0 w-5 h-6 flex items-center justify-center bg-transparent border-none text-muted cursor-pointer"
                  >
                    {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  </button>
                ) : <span className="shrink-0 w-5" />}
                {renaming ? (
                  <div className="flex-1 min-w-0 py-1">
                    <FolderNameInput
                      initial={folder.name}
                      placeholder={i18nT('pages.artifactsPage.rename_folder')}
                      onCommit={(name) => actions.onRenameSubmit(folder, name)}
                      onCancel={actions.onRenameCancel}
                    />
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => onSelect({ kind: 'folder', id: folder.id })}
                    aria-current={active ? 'page' : undefined}
                    className={`${ROW} flex-1 pl-0.5 ${active ? 'bg-transparent text-accent font-semibold' : ROW_IDLE}`}
                  >
                    <FolderGlyph folder={folder} size={15} />
                    <span className="truncate min-w-0 flex-1">{folder.name}</span>
                    <span className="text-[11px] text-muted font-normal tabular-nums group-hover:hidden group-focus-within:hidden">{counts.get(folder.id) ?? 0}</span>
                  </button>
                )}
                {!renaming && (
                  <div className="shrink-0 hidden group-hover:flex group-focus-within:flex [@media(hover:none)]:flex">
                    <FolderMenu folder={folder} folders={folders} actions={actions} />
                  </div>
                )}
              </div>
            )}
          </DndDraggable>
        )}
      </DndDroppable>
      {expanded && children.length > 0 && (
        <ul className="list-none m-0 p-0">
          {children.map((c) => (
            <FolderEntry
              key={c.id}
              folder={c}
              folders={folders}
              depth={depth + 1}
              selectedId={selectedId}
              expandedIds={expandedIds}
              onToggleExpand={onToggleExpand}
              onSelect={onSelect}
              actions={actions}
              counts={counts}
            />
          ))}
        </ul>
      )}
    </li>
  )
}

export function ArtifactsSideNav({
  selection, onSelect, folders, actions, expandedIds, onToggleExpand, totalCount, counts,
  creatingFolder, newFolderColor, onPickNewFolderColor, onCommitNewFolder, onCancelNewFolder, onNewFolder,
}: {
  selection: LibrarySelection
  onSelect: (sel: LibrarySelection) => void
  folders: ArtifactFolder[]
  actions: FolderActions
  expandedIds: ReadonlySet<string>
  onToggleExpand: (id: string) => void
  /** Every artifact in the library, for the "All artifacts" entry. */
  totalCount: number
  /** Artifacts filed directly in each folder. */
  counts: ReadonlyMap<string, number>
  creatingFolder: boolean
  newFolderColor: string
  onPickNewFolderColor: (color: string) => void
  onCommitNewFolder: (name: string) => void
  onCancelNewFolder: () => void
  onNewFolder: () => void
}) {
  const selectedId = selection.kind === 'folder' ? selection.id : ''
  const roots = childFolders(folders, '')
  return (
    <nav
      aria-label={i18nT('pages.artifactsPage.library_navigation')}
      className="w-[220px] shrink-0 border-r border-border overflow-y-auto px-2 py-4 flex flex-col gap-0.5"
    >
      <TopEntry
        label={i18nT('pages.artifactsPage.home')}
        icon={<House size={15} />}
        active={selection.kind === 'home'}
        onClick={() => onSelect({ kind: 'home' })}
      />
      <TopEntry
        label={i18nT('pages.artifactsPage.all_artifacts')}
        icon={<Library size={15} />}
        active={selection.kind === 'all'}
        count={totalCount}
        dropFolderId=""
        onClick={() => onSelect({ kind: 'all' })}
      />
      <PanelSectionHeader
        label={i18nT('pages.artifactsPage.folders')}
        className="mt-4 mb-1 px-2"
        trailing={(
          <button
            type="button"
            onClick={onNewFolder}
            aria-label={i18nT('pages.artifactsPage.new_folder')}
            title={i18nT('pages.artifactsPage.new_folder')}
            className="shrink-0 p-0.5 rounded bg-transparent border-none text-muted hover:text-text cursor-pointer"
          >
            <FolderPlus size={13} />
          </button>
        )}
      />
      <ul className="list-none m-0 p-0">
        {roots.map((f) => (
          <FolderEntry
            key={f.id}
            folder={f}
            folders={folders}
            depth={0}
            selectedId={selectedId}
            expandedIds={expandedIds}
            onToggleExpand={onToggleExpand}
            onSelect={onSelect}
            actions={actions}
            counts={counts}
          />
        ))}
      </ul>
      {creatingFolder && (
        <div className="px-2 py-1.5">
          <FolderNameInput
            placeholder={i18nT('pages.artifactsPage.new_folder_name')}
            onCommit={onCommitNewFolder}
            onCancel={onCancelNewFolder}
          />
          <div className="mt-1.5">
            <FolderColorSwatches size={13} value={newFolderColor} onPick={onPickNewFolderColor} />
          </div>
        </div>
      )}
      {roots.length === 0 && !creatingFolder && (
        <p className="px-2 py-1 text-[12px] text-muted m-0">{i18nT('pages.artifactsPage.no_folders_yet')}</p>
      )}
    </nav>
  )
}
