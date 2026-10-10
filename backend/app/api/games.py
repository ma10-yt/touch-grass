from fastapi import APIRouter, HTTPException
from typing import List
from app.models.game import (
    GameCreate,
    GameResponse,
    TargetChangeBallot,
    TargetChangeRequest,
)
from app.models.player import Player, PlayerJoin, PlayerResponse, LocationUpdate
from app.services.game_service import GameService, GameStartError
from app.ai.gemma_generator import GemmaModelUnavailable, TargetGenerationError
from app.ai.yolo_verifier import VerificationUnavailable
from app.services.location_service import LocationService
from app.services.target_change_service import TargetChangeError, TargetChangeService
from app.db.repositories.repositories import GameRepository, PlayerRepository
from app.config import settings

router = APIRouter()


@router.post("/{game_id}/players/{player_id}/heartbeat")
async def player_heartbeat(game_id: str, player_id: str):
    if not await PlayerRepository.touch_last_seen(game_id, player_id):
        raise HTTPException(status_code=404, detail="Player not found")
    return {"active": True}


@router.post("/{game_id}/targets/{target_id}/replacement-vote", response_model=GameResponse)
async def request_target_replacement(
    game_id: str,
    target_id: str,
    request: TargetChangeRequest,
):
    try:
        game = await TargetChangeService.request(game_id, request.player_id, target_id)
    except TargetChangeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return GameResponse.model_validate(game.model_dump(by_alias=True))


@router.post(
    "/{game_id}/target-change-vote/{proposal_id}",
    response_model=GameResponse,
)
async def cast_target_replacement_vote(
    game_id: str,
    proposal_id: str,
    ballot: TargetChangeBallot,
):
    try:
        game = await TargetChangeService.cast_vote(
            game_id,
            proposal_id,
            ballot.player_id,
            ballot.approve,
        )
    except TargetChangeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return GameResponse.model_validate(game.model_dump(by_alias=True))


@router.post("", response_model=GameResponse)
async def create_game(req: GameCreate):
    game = await GameService.create_game(req.creator_name, req.lat, req.lon)
    return GameResponse.model_validate(game.model_dump(by_alias=True))

@router.post("/join", response_model=PlayerResponse)
async def join_game(req: PlayerJoin):
    player = await GameService.join_game(req.join_code, req.display_name)
    if not player:
        raise HTTPException(status_code=400, detail="Invalid join code or game already started")
    return _player_response(player)

@router.post("/{game_id}/start")
async def start_game(game_id: str, creator_id: str):
    try:
        context = await GameService.start_game(game_id, creator_id)
    except GameStartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GemmaModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TargetGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VerificationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "message": "Game started",
        "context_status": context.context_status,
        "context_mode": context.context_mode,
        "environment_type": context.environment_type,
        "nearby_categories": context.nearby_categories,
        "context_provider": context.context_provider,
        "context_notice": context.notice,
    }

@router.post("/{game_id}/players/{player_id}/location")
async def update_location(game_id: str, player_id: str, loc: LocationUpdate):
    result = await LocationService.update_location(
        game_id,
        player_id,
        loc.lat,
        loc.lon,
        loc.accuracy_m,
        loc.timestamp,
    )
    if result.message == "Player not found in this game.":
        raise HTTPException(status_code=404, detail="Player not found")
    context_refreshed = False
    context_refresh_notice = None
    game = None
    if result.accepted:
        context_refreshed, context_refresh_notice = await GameService.refresh_area_context(
            game_id, loc.lat, loc.lon
        )
        game = await GameRepository.get_by_id(game_id)
    return {
        "accepted": result.accepted,
        "message": result.message,
        "movement_meters": round(result.movement_meters, 1),
        "movement_added_meters": round(result.movement_added_meters, 1),
        "movement_accepted": result.movement_accepted,
        "location_accuracy_m": loc.accuracy_m,
        "location_quality": (
            "higher-confidence"
            if loc.accuracy_m <= settings.GPS_HIGH_CONFIDENCE_METERS
            else "approximate"
        ),
        "location_context": game.location_context if game else None,
        "context_refreshed": context_refreshed,
        "context_refresh_notice": context_refresh_notice,
        "game_finished": result.game_finished,
    }

@router.get("/{game_id}", response_model=GameResponse)
async def get_game(game_id: str):
    game = await GameRepository.get_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameResponse.model_validate(game.model_dump(by_alias=True))

@router.get("/{game_id}/leaderboard", response_model=List[PlayerResponse])
async def get_leaderboard(game_id: str):
    players = await GameService.get_leaderboard(game_id)
    return [_player_response(p) for p in players]

@router.get("/{game_id}/player/{player_id}", response_model=PlayerResponse)
async def get_player(game_id: str, player_id: str):
    player = await PlayerRepository.get_by_id(player_id)
    if not player or player.game_id != game_id:
        raise HTTPException(status_code=404, detail="Player not found")
    return _player_response(player)


def _player_response(player: Player) -> PlayerResponse:
    payload = player.model_dump(by_alias=True)
    return PlayerResponse.model_validate(payload)
