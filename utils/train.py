import os
import torch
from tqdm import tqdm

def train(model, iterator, optimizer, criterion, device,
          metric_fn=None, grad_clip=None, grad_accum_steps=1):
    model.train()
    epoch_loss = 0
    steps = 0
    optimizer.zero_grad()
    metrics = None
    if metric_fn is not None:
        metrics = metric_fn()
    pbar = tqdm(iterator, leave=False)
    for i, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss = loss / grad_accum_steps
        loss.backward()
        if metrics:
            metrics.update(y_pred, y)
            current_metrics = {k: v.item() for k, v in metrics.compute().items()}
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in current_metrics.items()])
            if hasattr(pbar, 'set_description'):
                pbar.set_description(f"Loss: {loss.item()*grad_accum_steps:.4f}, {metrics_str}")
        elif hasattr(pbar, 'set_description'):
            pbar.set_description(f"Loss: {loss.item()*grad_accum_steps:.4f}")
        if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(iterator)):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
        epoch_loss += loss.item() * grad_accum_steps
    epoch_loss = epoch_loss / len(iterator)
    epoch_metrics = {}
    if metrics:
        epoch_metrics = {k: v.item() for k, v in metrics.compute().items()}
        metrics.reset()
    return epoch_loss, epoch_metrics

def evaluate(model, iterator, criterion, device, metric_fn=None):
    model.eval()
    epoch_loss = 0
    metrics = None
    if metric_fn is not None:
        metrics = metric_fn()
    with torch.no_grad():
        pbar = tqdm(iterator, leave=False)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if metrics:
                metrics.update(y_pred, y)
                current_metrics = {k: v.item() for k, v in metrics.compute().items()}
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in current_metrics.items()])
                if hasattr(pbar, 'set_description'):
                    pbar.set_description(f"Loss: {loss.item():.4f}, {metrics_str}")
            elif hasattr(pbar, 'set_description'):
                pbar.set_description(f"Loss: {loss.item():.4f}")
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(iterator)
    epoch_metrics = {}
    if metrics:
        epoch_metrics = {k: v.item() for k, v in metrics.compute().items()}
        metrics.reset()
    return epoch_loss, epoch_metrics

def run_training(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    num_epochs,
    metric_fn=None,
    scheduler=None,
    grad_clip=None,
    grad_accum_steps=1,
    early_stopping_patience=None,
    model_path=None,
    monitor_metric=None,
    logger=None
):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': []
    }
    best_val_value = float('inf')
    is_better = lambda new, best: new < best
    if monitor_metric:
        best_val_value = 0
        is_better = lambda new, best: new > best
    patience_counter = 0
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_loss, train_metrics = train(
            model, train_dataloader, optimizer, criterion,
            device, metric_fn, grad_clip, grad_accum_steps
        )
        if logger:
            metrics_dict = {'loss': train_loss}
            if train_metrics:
                metrics_dict.update(train_metrics)
            logger.log_metrics(metrics_dict, prefix='train')
        val_loss, val_metrics = evaluate(
            model, val_dataloader, criterion, device, metric_fn
        )
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        if logger:
            metrics_dict = {'loss': val_loss}
            if val_metrics:
                metrics_dict.update(val_metrics)
            logger.log_metrics(metrics_dict, prefix='val')
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_metrics({'learning_rate': current_lr})
            logger.step()
        metrics_str = ""
        if metric_fn:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            pbar.set_description(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | {metrics_str}")
        else:
            pbar.set_description(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        if monitor_metric and val_metrics:
            current_val_value = val_metrics.get(monitor_metric, val_loss)
        else:
            current_val_value = val_loss
        should_save = model_path and is_better(current_val_value, best_val_value)
        should_stop = early_stopping_patience and patience_counter >= early_stopping_patience
        if should_save:
            best_val_value = current_val_value
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        elif early_stopping_patience:
            patience_counter += 1
            if should_stop:
                break
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return history